"""Excepciones de dominio del gateway."""


class StoreWithoutBifrostKeyError(Exception):
    """La tienda resuelta no tiene una VK de Bifrost activa.

    Indica un error de creación/renovación de claves en el panel (toda tienda
    activa en chatwoot_account debe tener una `bifrost_key`). El endpoint la
    traduce a HTTP 503 y eia-bot responde con un mensaje amigable al cliente.
    """

    def __init__(self, account_id: int | None, store_name: str):
        self.account_id = account_id
        self.store_name = store_name
        super().__init__(
            f"La tienda '{store_name}' (account_id={account_id}) no tiene "
            "una clave Bifrost activa"
        )
