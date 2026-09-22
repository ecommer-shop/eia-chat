from dataclasses import dataclass, field


@dataclass
class StoreConfig:
    store_name: str
    account_id: int | None = None
    channel_name: str = ""
    channel_tokens: list[str] = field(default_factory=list)
    is_global: bool = False
    audience: str = "CLIENTE"
    system_prompt: str = ""
    language: str = "es"
    is_mapped: bool = True
    few_shot: list[dict] = field(default_factory=list)
    inbox_map: dict[int, str] = field(default_factory=dict)
    prompts: dict[str, str] = field(default_factory=dict)