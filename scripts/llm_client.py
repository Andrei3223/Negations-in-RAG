"""
Shared Azure OpenAI client utilities.

Used by:
  scripts/llm_judge_eval.py
  scripts/gen_data/generate_answers.py
"""

import os

from openai import AzureOpenAI


def make_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_KEY"],
        api_version=os.environ["OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        timeout=30.0,
    )


def call_with_retry(
    client: AzureOpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 30,
    temperature: float = 0,
    retries: int = 3,
) -> str | None:
    """Call chat completions with retry logic. Returns stripped response text or None on failure."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            if text:
                return text
        except Exception as e:
            if attempt == retries - 1:
                print(f"    WARNING: API error after {retries} retries: {e}")
    return None
