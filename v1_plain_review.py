"""
v1 — Code Review Agent (plain script, no LangGraph, no tools)
Paste code -> get a Gemini-generated review based purely on reading the text.
No execution, no verification — this is the baseline to compare later
agentic versions against.
"""

from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

REVIEW_PROMPT = """You are an experienced senior software engineer conducting a code review.
Review the following code and provide feedback covering:
1. Bugs or potential errors
2. Code style / readability issues
3. Performance concerns (if any)
4. Suggestions for improvement

Be specific — reference line numbers or exact code snippets where relevant.
If the code is already good, say so, don't invent problems.

Code to review:
```
{code}
```
"""


def get_review(code: str) -> str:
    prompt = REVIEW_PROMPT.format(code=code)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    print("Paste your code below. Type 'END' on a new line when done:\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    code = "\n".join(lines)

    if not code.strip():
        print("No code entered.")
    else:
        print("\n--- Reviewing your code... ---\n")
        review = get_review(code)
        print(review)
