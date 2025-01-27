from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


class Module:
    """ 模块构成了一个树状结构，用于存储参数和其他子模块。它们是神经网络堆栈的基础。
    
    Attributes
        _modules ：子模块
        _parameters ：模块参数
        training ：训练模式 or 评估模式
    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        """Return the direct child modules of this module."""
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        """ 把模块树中全部模块设置为训练模式 """        
        self.training = True
        for module in self.modules():
            module.train()

    def eval(self) -> None:
        """ 把模块树中全部模块设置为评估模式 """   
        self.training = False
        for module in self.modules():
            module.eval()

    def named_parameters_helper(self, module_name: str) :
        named_params = []
        
        for name, param in self._parameters.items():
            full_name = module_name + name
            named_params.append((full_name, param))

        for name, module in self._modules.items():
            full_name = module_name + name + '.'
            named_params.extend(module.named_parameters_helper(full_name))

        return named_params

    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        return self.named_parameters_helper("")


    def parameters(self) -> Sequence[Parameter]:
        """ 返回当前模块及其所有子模块中的所有参数 """
        current_params = list(self._parameters.values())

        for sub_module in self._modules.values():
            current_params.extend(sub_module.parameters())

        return current_params

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """Manually add a parameter. Useful helper for scalar parameters.

        Args:
        ----
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
        -------
            Newly created parameter.

        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """A Parameter is a special container stored in a `Module`.

    It is designed to hold a `Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        """Update the parameter value."""
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
