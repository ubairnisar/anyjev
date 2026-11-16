"""anyjev - logprob-scored LLM judgements on any model served by vLLM."""

from . import presets
from .calibrate import Calibration, calibrate_threshold
from .judge import Judge
from .result import OrderScore, Result
from .scoring import LabelNotFoundError
from .task import Option, Task, binary

__version__ = "0.1.0"

__all__ = [
    "Judge",
    "Task",
    "Option",
    "binary",
    "Result",
    "OrderScore",
    "Calibration",
    "calibrate_threshold",
    "LabelNotFoundError",
    "presets",
    "__version__",
]
