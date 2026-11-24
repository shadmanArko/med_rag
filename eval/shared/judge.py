from __future__ import annotations

import json
import os
import re

from deepeval.models import DeepEvalBaseLLM

DEFAULT_JUDGE_MODEL = "claude-haiku-5-5"


class ClaudeJudge(DeepEvalBaseLLM):
    """DeepEval judge model backed by the Anthropic API (replaces the OpenAI default)."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_JUDGE_MODEL)
        super().__init__(self.model_name)

    def load_model(self):
        from anthropic import Anthropic, AsyncAnthropic

        return Anthropic(), AsyncAnthropic()

    @staticmethod
    def _prompt(prompt: str, schema) -> str:
        if schema is None:
            return prompt
        return (
            f"{prompt}\n\nRespond with ONLY a valid JSON object matching this JSON schema, "
            f"no prose and no code fences:\n{json.dumps(schema.model_json_schema())}"
        )

    @staticmethod
    def _parse(text: str, schema):
        if schema is None:
            return text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return schema(**json.loads(match.group(0) if match else text))

    def generate(self, prompt: str, schema=None):
        client, _ = self.load_model()
        message = client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": self._prompt(prompt, schema)}],
        )
        return self._parse(message.content[0].text, schema)

    async def a_generate(self, prompt: str, schema=None):
        _, client = self.load_model()
        message = await client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            messages=[{"role": "user", "content": self._prompt(prompt, schema)}],
        )
        return self._parse(message.content[0].text, schema)

    def get_model_name(self) -> str:
        return self.model_name
