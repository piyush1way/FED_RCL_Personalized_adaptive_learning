from typing import Any, Dict, Iterable, Iterator, Tuple
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

class Registry(Iterable[Tuple[str, Any]]):
    """
    A flexible registry that maps names to objects for easy model/configuration management.

    Example usage:
    
    ```python
    BACKBONE_REGISTRY = Registry('BACKBONE')

    @BACKBONE_REGISTRY.register()
    class MyBackbone:
        ...
    
    # OR
    BACKBONE_REGISTRY.register(MyBackbone)
    ```
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): The name of this registry.
        """
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}

    def _do_register(self, name: str, obj: Any) -> None:
        """
        Internal function to register an object by name.

        Args:
            name (str): The name to register the object under.
            obj (Any): The object to register.

        Raises:
            ValueError: If an object with the same name is already registered.
        """
        if name in self._obj_map:
            logger.warning(f"⚠️ Warning: '{name}' is already registered in '{self._name}' registry. Skipping.")
            return  # ✅ Prevents duplicate registration without crashing.

        self._obj_map[name] = obj

    def register(self, obj: Any = None) -> Any:
        """
        Register an object in the registry.
        Can be used as a decorator or function call.

        Example usage:
        ```python
        @MY_REGISTRY.register()
        class MyClass:
            ...
        ```
        
        OR

        ```python
        MY_REGISTRY.register(MyClass)
        ```

        Returns:
            The registered object for further use.
        """

        if obj is None:
            # Used as a decorator
            def decorator(func_or_class: Any) -> Any:
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return decorator

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj)
        return obj  # ✅ Ensures function call returns the object

    def get(self, name: str) -> Any:
        """
        Retrieve an object by name from the registry.

        Args:
            name (str): The registered name of the object.

        Returns:
            Any: The registered object.

        Raises:
            KeyError: If the name is not found.
        """
        if name not in self._obj_map:
            raise KeyError(
                f"❌ No object named '{name}' found in '{self._name}' registry! "
                f"Available options: {list(self._obj_map.keys())}"
            )
        return self._obj_map[name]

    def __contains__(self, name: str) -> bool:
        """Check if a name exists in the registry."""
        return name in self._obj_map

    def __repr__(self) -> str:
        """Display the registered objects in a table format."""
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return f"📌 Registry of {self._name}:\n{table}"

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Make the registry iterable."""
        return iter(self._obj_map.items())

    __str__ = __repr__  # Make __str__ behave like __repr__
