import logging
from typing import Dict, List

import jax.numpy as jnp
import numpy as np
import flax
from flax.struct import field  # Used for default_factory in dataclasses


# --- Metric Definitions (Replaces clu.metrics) ---
@flax.struct.dataclass
class TrainMetrics:
    loss: List[float] = field(default_factory=list)
    kld_loss: List[float] = field(default_factory=list)
    mae_loss: List[float] = field(default_factory=list)
    ssim_loss: List[float] = field(default_factory=list)
    percept_loss: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)

    @classmethod
    def empty(cls):
        return cls(
            loss=[],
            kld_loss=[],
            mae_loss=[],
            ssim_loss=[],
            percept_loss=[],
            learning_rate=[],
        )

    def update(self, **kwargs) -> "TrainMetrics":
        updates = {}
        current_fields = self.__annotations__.keys()
        for key, value in kwargs.items():
            if key in current_fields:
                current_list = getattr(self, key)
                value_to_append = (
                    value.item()
                    if isinstance(value, (jnp.ndarray, np.ndarray)) and value.ndim == 0
                    else value
                )
                updates[key] = current_list + [value_to_append]
            else:
                logging.debug(
                    f"Metric key '{key}' not found in TrainMetrics, ignoring."
                )
        return self.replace(**updates)

    def compute(self) -> Dict[str, float]:
        metrics_dict = {}
        for metric_name in self.__annotations__.keys():
            values = getattr(self, metric_name)
            if values:
                if metric_name == "learning_rate":
                    metrics_dict[metric_name] = values[-1]
                else:
                    metrics_dict[metric_name] = np.mean(values)
        return metrics_dict


@flax.struct.dataclass
class EvalMetrics:
    loss: List[float] = field(default_factory=list)
    kld_loss: List[float] = field(default_factory=list)
    mae_loss: List[float] = field(default_factory=list)
    ssim_loss: List[float] = field(default_factory=list)
    percept_loss: List[float] = field(default_factory=list)

    @classmethod
    def empty(cls):
        return cls(
            loss=[],
            kld_loss=[],
            mae_loss=[],
            ssim_loss=[],
            percept_loss=[],
        )

    def update(self, **kwargs) -> "EvalMetrics":
        updates = {}
        current_fields = self.__annotations__.keys()
        for key, value in kwargs.items():
            if key in current_fields:
                current_list = getattr(self, key)
                value_to_append = (
                    value.item()
                    if isinstance(value, (jnp.ndarray, np.ndarray)) and value.ndim == 0
                    else value
                )
                updates[key] = current_list + [value_to_append]
            else:
                logging.debug(f"Metric key '{key}' not found in EvalMetrics, ignoring.")
        return self.replace(**updates)

    def compute(self) -> Dict[str, float]:
        metrics_dict = {}
        for metric_name in self.__annotations__.keys():
            values = getattr(self, metric_name)
            if values:
                metrics_dict[metric_name] = np.mean(values)
        return metrics_dict
