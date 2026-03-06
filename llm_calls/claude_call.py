import os
import re
import json
import time
from typing import Optional, Union, Dict, List

from anthropic import Anthropic, APIStatusError

# ---- Config ----
ANTHROPIC_MODEL_DEFAULT = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

client = Anthropic(api_key=API_KEY)

def claude_call_for_code(
    system_prompt: str,
    user_prompt: str,
    content = None,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    temperature: float = 0.0,
) -> str:
    """
    Call Anthropic Claude to generate Python code only.
    Cleans the response by extracting the last fenced code block if present.
    """

    def _clean_code(text: str) -> str:
        text = text.strip()
        # Prefer the LAST fenced code block (``` or ```python)
        blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
        if blocks:
            return blocks[-1].strip()
        # Fallback: trim leading non-code until code-ish line
        lines = text.splitlines()
        pat = re.compile(r'^\s*(from\s+\w+|import\s+\w+|def\s+\w+|class\s+\w+|#|"""|\'\'\')')
        for i, ln in enumerate(lines):
            if pat.search(ln):
                return "\n".join(lines[i:]).strip()
        return text

    attempt = 0
    while attempt < max_retries:
        try:
            contents = user_prompt
            if content is not None:
                content_str = json.dumps(content, indent=2) if isinstance(content, (dict, list)) else str(content)
                contents += f"\n\n--- Additional Context ---\n{content_str}"

            resp = client.messages.create(
                model=ANTHROPIC_MODEL_DEFAULT,
                system=system_prompt,
                temperature=0,
                max_tokens=4096,  # adjust as needed
                messages=[
                    {"role": "user", "content": contents}
                ],
            )

            # Anthropic returns a list of content blocks; concatenate text blocks
            if not resp.content:
                raise ValueError("Empty response content from Claude.")
            raw_text = "".join(
                part.text for part in resp.content if getattr(part, "type", "text") == "text"
            ).strip()
            if not raw_text:
                raise ValueError("No text returned by Claude.")
            return _clean_code(raw_text)

        except (APIStatusError, Exception) as e:
            attempt += 1
            print(f"[Retry {attempt}/{max_retries}] Claude API call failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                raise RuntimeError(f"Claude API call failed after {max_retries} retries.") from e


# --- Quick self-test (mirrors your Gemini test) ---
def test_claude_call_for_code():
    system_prompt = "You are a Python code generator. Respond ONLY with executable Python code."
    user_prompt = (
        "Write Python code that creates a pandas DataFrame with two columns A and B, "
        "each containing numbers 1 to 5, and prints the sum of column A."
    )
    try:
        code = claude_call_for_code(system_prompt, user_prompt)
        print("\n--- Cleaned Code from Claude ---\n")
        print(code)

        # Optional: execute the returned code (sandbox appropriately in real apps)
        exec_globals = {}
        exec(code, exec_globals)
        print("\n--- Execution Completed ---\n")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    test_claude_call_for_code()
