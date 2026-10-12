"""Loading a script from ``scripts/`` as a module, so a test can call its functions.

``scripts/`` is not a package and is not on the import path: a script is a command, and
only its functions are worth testing.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> ModuleType:
    """The module in ``scripts/<name>.py``, imported once and kept under ``scripts.<name>``."""
    qualified = f"scripts.{name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(qualified, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module
