# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# pyre-ignore-all-errors[2,3]
from typing import Any, Dict, Iterable, Iterator, Tuple
from tabulate import tabulate


class Registry(Iterable[Tuple[str, Any]]):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        if name in self._obj_map:
            raise ValueError(
                f"An object named '{name}' is already registered in '{self._name}' registry!"
            )
        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """
        Register the given object under the name `obj.__name__`.
        Can be used as either a decorator or function call.

        Returns:
            The registered object for chaining.
        """

        if obj is None:
            # used as a decorator
            def decorator(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return decorator

        # used as a function call
        name = obj.__name__
        self._do_register(name, obj)
        return obj  # ✅ Ensure function call returns the object

    def get(self, name: str) -> Any:
        """
        Retrieve an object by name from the registry.

        Raises:
            KeyError if the name is not found.
        """
        if name not in self._obj_map:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return f"Registry of {self._name}:\n{table}"

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())

    __str__ = __repr__  # Make __str__ behave like __repr__
