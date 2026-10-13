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
    # VK de Bifrost usada por los LLM de esta tienda (por plan de pago).
    # `repr=False`: los dataclass NO deben exponerla en logs ni en las trazas
    # de Langfuse (serializa los argumentos de las funciones observadas).
    bifrost_vk: str | None = field(default=None, repr=False)
    # Plan de pago asociado a la VK (free | tienda | omnichannel | ecommer).
    bifrost_plan: str = ""