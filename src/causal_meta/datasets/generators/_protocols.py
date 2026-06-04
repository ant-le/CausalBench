from __future__ import annotations

from typing import Any, Protocol


class Instantiable(Protocol):
    """Protocol for objects that can create instances via ``instantiate()``."""

    def instantiate(self) -> Any: ...
