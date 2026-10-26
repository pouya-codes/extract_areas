"""
AI Models Package

This package contains all AI processing models for the platform.

Concrete models are imported lazily: each worker image ships only the
dependencies of the model it serves, so importing ``models`` must not drag
in DeepLIIF, HoVer-Net or GigaPath for a container that has none of them.
``from models import DeepLIIFModel`` still works; it just resolves on use.
"""

from models.base_model import BaseAIModel
from models.model_registry import ModelRegistry, model_registry

_LAZY = {
    'DeepLIIFModel': ('models.deepliif_model', 'DeepLIIFModel'),
    'ExampleModel': ('models.example_model', 'ExampleModel'),
    'ECCancerModel': ('models.ec_cancer_model', 'ECCancerModel'),
    'HoVerNetModel': ('models.hovernet_model', 'HoVerNetModel'),
    'PDACModel': ('models.pdac_model', 'PDACModel'),
    'PatchClassifierModel': ('models.patch_classifier_model', 'PatchClassifierModel'),
}


def __getattr__(name):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'models' has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), attr)


__all__ = [
    'BaseAIModel',
    'ModelRegistry',
    'model_registry',
    *_LAZY.keys(),
]
