from typing import Any, Dict, Optional, Protocol


class BaseLogger(Protocol):
    """
    Interface for experiment loggers (e.g., WandB, Local/Console).
    """

    @property
    def run_id(self) -> Optional[str]:
        """
        Return the logger's run ID for checkpoint persistence.

        Returns None for loggers that don't support run resumption.
        """
        ...

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """
        Log a dictionary of scalar metrics.
        """
        ...

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """
        Log configuration/hyperparameters.
        """
        ...

    def finish(self) -> None:
        """
        Clean up resources (e.g., close connections, save files).
        """
        ...
