_registry = {}

def register_kernel(cls):
    if not hasattr(cls, "_kernel_name"):
        raise ValueError(f"Class {cls.__name__} must define '_kernel_name'")
    _registry[cls._kernel_name] = cls
    return cls

def get_kernel_class(name: str) -> type:
    if name not in _registry:
        raise KeyError(f"No kernel registered for '{name}'. Available: {list(_registry.keys())}")
    return _registry[name]
