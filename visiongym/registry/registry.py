from typing import Dict, Any, Tuple, Iterator
from tabulate import tabulate


class Registry:
    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: Dict[str, Any] = {}
        
    def _reg(self, name: str, obj: Any) -> None:
        assert name not in self._obj_map, f"object '{name}' already registered"
        
        self._obj_map[name] = obj
        
    def register(self, obj: Any = None) -> Any:
        if obj is None:
            # when used as a decorator
            def decorator_subfunction(cls: Any) -> Any:
                name = cls.__name__
                self._reg(name, cls)
                return cls
            
            return decorator_subfunction
       
       # manually register an object 
        name = obj.__name__
        self._reg(name, obj)
    
    def get(self, name: str) -> Any:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(name, self._name)
            )
        return ret
    
    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __repr__(self) -> str:
        table_headers = ["Names", "Objects"]
        table = tabulate(
            self._obj_map.items(), headers=table_headers, tablefmt="fancy_grid"
        )
        return "Registry of {}:\n".format(self._name) + table

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return iter(self._obj_map.items())
