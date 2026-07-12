"""
v2 — Code Review Agent (ReAct agent, ONE tool: run_code)
First agentic version. Adds a real tool (run_code, via subprocess) so the
agent can actually execute code instead of just reading it. No structured
review format yet — that comes in v3.
"""

from google.genai import types
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from google import genai
import tempfile
from langgraph.graph import StateGraph, END

import operator
import os
import subprocess

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

run_code_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="run_code",
            description="Executes a code snippet in the specified language and returns the output or error.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "code": types.Schema(type="STRING", description="The code to execute"),
                    "language": types.Schema(
                        type="STRING",
                        description="The programming language of the code, e.g. 'python', 'cpp', 'javascript'"
                    )
                },
                required=["code", "language"]
            )
        )
    ]
)


def run_code(code: str, language: str) -> str:
    language = language.lower().strip()

    ext_map = {
        "python": ".py",
        "cpp": ".cpp",
        "c++": ".cpp",
        "javascript": ".js",
        "js": ".js",
    }

    extension = ext_map.get(language)
    if extension is None:
        return f"Unsupported language: {language}"

    temp_dir = tempfile.gettempdir()
    full_filename = f"{temp_dir}/temp_code{extension}"

    with open(full_filename, 'w', encoding='utf-8') as file:
        file.write(code)

    if extension == '.py':
        command = ["python", full_filename]
    elif extension == '.js':
        command = ["node", full_filename]
    elif extension == '.cpp':
        exe_path = full_filename.replace('.cpp', '.exe')
        compile_result = subprocess.run(
            ["g++", full_filename, "-o", exe_path],
            capture_output=True, text=True, encoding='utf-8', timeout=10
        )
        if compile_result.returncode != 0:
            os.remove(full_filename)
            return f"Compilation failed:\n{compile_result.stderr}"
        command = [exe_path]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=10
        )

        if result.returncode == 0:
            output = result.stdout or "Code ran successfully with no output."
        else:
            output = f"Code ran but exited with an error:\n{result.stderr}"

    except subprocess.TimeoutExpired:
        output = "Execution timed out (possible infinite loop)."

    except FileNotFoundError:
        output = f"Could not run — is the interpreter/compiler for {language} installed?"

    finally:
        if os.path.exists(full_filename):
            os.remove(full_filename)

    return output


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


config = types.GenerateContentConfig(tools=[run_code_tool])


def think(state: AgentState) -> dict:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=state["messages"],
        config=config
    )
    return {"messages": [response.candidates[0].content]}


def execute_tool(state: AgentState) -> dict:
    last_content = state["messages"][-1]

    for part in last_content.parts:
        if part.function_call:
            result = run_code(
                part.function_call.args["code"],
                part.function_call.args["language"]
            )

            tool_response = types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name="run_code",
                    response={"result": result}
                )]
            )
            return {"messages": [tool_response]}


def should_continue(state: AgentState) -> str:
    last_content = state["messages"][-1]
    for part in last_content.parts:
        if part.function_call:
            return "execute_tool"
    return "end"


graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_node("execute_tool", execute_tool)
graph.set_entry_point("think")

graph.add_conditional_edges(
    "think",
    should_continue,
    {"execute_tool": "execute_tool", "end": END}
)
graph.add_edge("execute_tool", "think")

app = graph.compile()

if __name__ == "__main__":
    initial_state = {
        "messages": [
            types.Content(role="user", parts=[types.Part(
                text="""Run this code and tell me the output:
for i in range(5):
    print(i * i)""")])
        ]
    }

    result = app.invoke(initial_state)

    for msg in result["messages"]:
        for part in msg.parts:
            if part.text:
                print("TEXT:", part.text)
            if part.function_call:
                print("TOOL CALL:", part.function_call.name, part.function_call.args)
            if part.function_response:
                print("TOOL RESULT:", part.function_response.response)
