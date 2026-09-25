"""Model registry: each classifier self-registers with @register in its own file.

To add a model: create a new file in this package exposing a `build(random_state)`
function decorated with `@register("your_model_name")`. Nothing else needs to
change here or in `get_models` — the file is picked up automatically.
"""

import importlib
import pkgutil

_REGISTRY = {}


def register(name: str):
    """Decorator: register a `build(random_state) -> estimator` function under `name`."""

    def decorator(build_fn):
        _REGISTRY[name] = build_fn
        return build_fn

    return decorator


def get_models(random_state: int = 42) -> dict:
    """Return name -> unfitted, sklearn-compatible classifier for every registered model.

    All models share `random_state` for comparability.
    """
    _discover()
    return {
        name: build_fn(random_state=random_state)
        for name, build_fn in _REGISTRY.items()
    }


def _discover():
    package = importlib.import_module(__package__)
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        if module_name != "registry":
            importlib.import_module(f"{__package__}.{module_name}")
