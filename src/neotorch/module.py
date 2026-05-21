"""Module and parameter abstractions for Neotorch models."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from ._tensor import Tensor  # pyright: ignore[reportMissingModuleSource]


def _validate_name(name: str | None) -> str | None:
    if name is None:
        return None
    if not isinstance(name, str):
        raise TypeError("name must be a string or None")
    if name == "":
        raise ValueError("name must be non-empty")
    if "." in name:
        raise ValueError("name must not contain '.'")
    return name


class Parameter(Tensor):
    """Tensor subclass used to mark trainable module state.

    A parameter shares the normal Tensor API but is registered by Module when it
    is assigned as a public attribute. The optional ``name`` metadata overrides
    the attribute segment used by ``Module.get_named_parameters``.

    Args:
        tensor_or_data: Existing Tensor to wrap as a Parameter, or backing data
            object for direct Tensor construction.
        offset: Data offset used when constructing from backing data.
        layout: Layout used when constructing from backing data.
        name: Optional public name metadata. When provided, it must be a
            non-empty string without ``.`` and is used as this parameter's
            segment in named-parameter traversal.

    Returns:
        Parameter instance backed by the provided tensor storage or data, with
        mutable ``name`` metadata.

    Examples:
        >>> from neotorch import Generic, Layout, Parameter, Shape, Stride, Tensor
        >>> tensor = Tensor(Generic([1]), 0, Layout(Shape(1), Stride(1)))
        >>> parameter = Parameter(tensor, name="weight")
        >>> parameter[0]
        1
        >>> parameter.name
        'weight'
    """

    def __init__(
        self,
        tensor_or_data: Tensor | Any,
        offset: int | None = None,
        layout: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        if isinstance(tensor_or_data, Tensor):
            if offset is not None or layout is not None:
                raise TypeError(
                    "Parameter constructed from a Tensor does not accept "
                    "offset or layout"
                )
            super().__init__(
                tensor_or_data.data, tensor_or_data.offset, tensor_or_data.layout
            )
            self.name = _validate_name(name)
            return

        if offset is None or layout is None:
            raise TypeError(
                "Parameter requires offset and layout when constructed from data"
            )
        super().__init__(tensor_or_data, offset, layout)
        self.name = _validate_name(name)


class Module:
    """Base class for Neotorch modules with parameter registration.

    Module subclasses implement ``forward``. Calling a module invokes
    ``forward`` and public Parameter or Module attributes are automatically
    registered for recursive discovery.

    Args:
        name: Optional public name metadata. When this module is assigned as a
            child module, the name overrides the attribute segment used by
            ``get_named_parameters``. It must be a non-empty string without
            ``.``.

    Returns:
        Module instance with empty parameter and submodule registries and
        mutable ``name`` metadata.

    Examples:
        >>> import neotorch
        >>> class Scale(neotorch.Module):
        ...     def __init__(self):
        ...         super().__init__(name="scale")
        ...         self.weight = neotorch.Parameter(
        ...             neotorch.Tensor(
        ...                 neotorch.Generic([2]),
        ...                 0,
        ...                 neotorch.Layout(neotorch.Shape(1), neotorch.Stride(1)),
        ...             ),
        ...             name="factor",
        ...         )
        ...     def forward(self, tensor):
        ...         return tensor * self.weight
        >>> root = neotorch.Module()
        >>> root.layer = Scale()
        >>> root.get_named_parameters()[0][0]
        'scale.factor'
    """

    name: str | None
    _parameters: dict[str, Parameter]
    _modules: dict[str, Module]

    def __init__(self, *, name: str | None = None) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "name", _validate_name(name))

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or not self._registries_are_ready():
            object.__setattr__(self, name, value)
            return
        if name == "name":
            object.__setattr__(self, name, _validate_name(value))
            return

        parameters = self._parameters
        modules = self._modules
        if isinstance(value, Parameter):
            parameters[name] = value
            modules.pop(name, None)
        elif isinstance(value, Module):
            modules[name] = value
            parameters.pop(name, None)
        else:
            parameters.pop(name, None)
            modules.pop(name, None)

        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if self._registries_are_ready():
            self._parameters.pop(name, None)
            self._modules.pop(name, None)
        object.__delattr__(self, name)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Compute the module output.

        Subclasses override this method with model-specific computation.

        Args:
            *args: Positional inputs for the module computation.
            **kwargs: Keyword inputs for the module computation.

        Returns:
            The module output.

        Examples:
            >>> import neotorch
            >>> class Identity(neotorch.Module):
            ...     def forward(self, value):
            ...         return value
            >>> Identity().forward("x")
            'x'
        """

        raise NotImplementedError("Module subclasses must implement forward")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke ``forward`` with the provided inputs.

        Args:
            *args: Positional inputs forwarded to ``forward``.
            **kwargs: Keyword inputs forwarded to ``forward``.

        Returns:
            The value returned by ``forward``.

        Examples:
            >>> import neotorch
            >>> class Identity(neotorch.Module):
            ...     def forward(self, value):
            ...         return value
            >>> Identity()("x")
            'x'
        """

        return self.forward(*args, **kwargs)

    def modules(self) -> tuple[Module, ...]:
        """Return this module and all recursive submodules.

        Traversal is deterministic, starts with ``self``, and de-duplicates
        shared submodule instances.

        Args:
            None.

        Returns:
            Tuple containing this module followed by recursive submodules.

        Examples:
            >>> import neotorch
            >>> root = neotorch.Module()
            >>> root.child = neotorch.Module()
            >>> root.modules() == (root, root.child)
            True
        """

        return tuple(self._iter_modules(set()))

    def parameters(self) -> tuple[Parameter, ...]:
        """Return all recursive parameters.

        Parameters are yielded in deterministic module traversal order and
        shared Parameter instances are returned once.

        Args:
            None.

        Returns:
            Tuple of recursive Parameter objects.

        Examples:
            >>> import neotorch
            >>> root = neotorch.Module()
            >>> root.weight = neotorch.Parameter(
            ...     neotorch.Tensor(
            ...         neotorch.Generic([1]),
            ...         0,
            ...         neotorch.Layout(neotorch.Shape(1), neotorch.Stride(1)),
            ...     )
            ... )
            >>> root.parameters() == (root.weight,)
            True
        """

        return tuple(
            parameter
            for _name, parameter in self._iter_named_parameters(set(), set(), "")
        )

    def get_named_parameters(self) -> tuple[tuple[str, Parameter], ...]:
        """Return recursive parameters with qualified names.

        Names use ``.`` separators for submodules and shared Parameter instances
        are returned only at their first traversal path.

        Args:
            None.

        Returns:
            Tuple of ``(qualified_name, Parameter)`` pairs.

        Examples:
            >>> import neotorch
            >>> root = neotorch.Module()
            >>> root.child = neotorch.Module()
            >>> root.child.weight = neotorch.Parameter(
            ...     neotorch.Tensor(
            ...         neotorch.Generic([1]),
            ...         0,
            ...         neotorch.Layout(neotorch.Shape(1), neotorch.Stride(1)),
            ...     )
            ... )
            >>> root.get_named_parameters()[0][0]
            'child.weight'
        """

        return tuple(self._iter_named_parameters(set(), set(), ""))

    def _registries_are_ready(self) -> bool:
        return "_parameters" in self.__dict__ and "_modules" in self.__dict__

    def _iter_modules(self, seen: set[int]) -> Iterator[Module]:
        module_id = id(self)
        if module_id in seen:
            return
        seen.add(module_id)
        yield self
        for module in self._modules.values():
            yield from module._iter_modules(seen)

    def _iter_named_parameters(
        self,
        seen_parameters: set[int],
        seen_modules: set[int],
        prefix: str,
    ) -> Iterator[tuple[str, Parameter]]:
        module_id = id(self)
        if module_id in seen_modules:
            return
        seen_modules.add(module_id)

        for name, parameter in self._parameters.items():
            parameter_id = id(parameter)
            if parameter_id in seen_parameters:
                continue
            seen_parameters.add(parameter_id)
            yield f"{prefix}{self._parameter_name(name, parameter)}", parameter

        for name, module in self._modules.items():
            yield from module._iter_named_parameters(
                seen_parameters,
                seen_modules,
                f"{prefix}{self._module_name(name, module)}.",
            )

    @staticmethod
    def _parameter_name(attribute_name: str, parameter: Parameter) -> str:
        return parameter.name or attribute_name

    @staticmethod
    def _module_name(attribute_name: str, module: Module) -> str:
        return module.name or attribute_name


__all__ = [
    "Module",
    "Parameter",
]
