import os
from dataclasses import dataclass
from typing import Any, Optional


DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.2
    timeout: int = 180


class OpenAICompatibleClient:
    supports_image_input = True

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        if not config.api_key:
            raise ValueError("API key is required. Pass an API key argument or set OPENAI_API_KEY/API_KEY.")
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url.rstrip("/"))
        self.model = config.model
        self.temperature = config.temperature

    def get_completion(
        self,
        system_prompt: str,
        prompt: str,
        seed: int = 42,
        image_urls: Optional[list[str]] = None,
    ) -> Optional[str]:
        image_urls = image_urls or []
        content: str | list[dict[str, Any]]
        if image_urls:
            content = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": url}} for url in image_urls)
        else:
            content = prompt

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
        }
        try:
            kwargs["seed"] = seed
            completion = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("seed", None)
            completion = self.client.chat.completions.create(**kwargs)
        return str(completion.choices[0].message.content or "")


def resolve_api_key(args, provider_key: str = "") -> str:
    cli_value = ""
    if provider_key:
        cli_value = getattr(args, provider_key, "") or ""
    cli_value = cli_value or getattr(args, "apikey", "") or getattr(args, "sk", "")
    if cli_value:
        return cli_value
    for env_name in ("OPENAI_API_KEY", "API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    return ""


def resolve_base_url(args, provider_key: str = "") -> str:
    cli_value = ""
    if provider_key:
        cli_value = getattr(args, provider_key, "") or ""
    return cli_value or getattr(args, "base_url", "") or os.getenv("OPENAI_BASE_URL") or os.getenv("BASE_URL") or DEFAULT_BASE_URL


def model_config_from_args(args, purpose: str = "generation") -> LLMConfig:
    provider = getattr(args, "model", "openai")
    if provider == "openai":
        model = getattr(args, "openai_model", "gpt-4o")
        api_key_name = "openai_apikey"
        base_url_name = "openai_base_url"
    elif provider == "gemini":
        model = getattr(args, "gemini_model", "gemini-2.0-flash")
        api_key_name = "gemini_apikey"
        base_url_name = "gemini_base_url"
    elif provider == "deepseek":
        model = getattr(args, "deepseek_model", "deepseek-chat")
        api_key_name = "deepseek_apikey"
        base_url_name = "deepseek_base_url"
    elif provider == "claude":
        model = getattr(args, "claude_model", "claude-3-7-sonnet-20250219")
        api_key_name = "claude_apikey"
        base_url_name = "claude_base_url"
    else:
        model = getattr(args, "openai_model", provider)
        api_key_name = "openai_apikey"
        base_url_name = "openai_base_url"

    return LLMConfig(
        provider=provider,
        model=model,
        api_key=resolve_api_key(args, api_key_name),
        base_url=resolve_base_url(args, base_url_name),
        temperature=getattr(args, "temperature", 0.2 if purpose == "generation" else 0.0),
        timeout=getattr(args, "timeout", 180),
    )


class LLMHandler:
    def __init__(self, args, enabled: bool = True, purpose: str = "generation"):
        if not enabled:
            self.handler = None
            self.model_name = "no_llm"
            return
        self.config = model_config_from_args(args, purpose=purpose)
        self.handler = OpenAICompatibleClient(self.config)
        self.model_name = self.config.model
