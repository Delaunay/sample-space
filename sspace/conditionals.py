from dataclasses import dataclass
from typing import Union


class _Condition:
    def visit(self, visitor, mode, *args, **kwargs):
        raise NotImplementedError()


@dataclass
class _LeafCondition(_Condition):
    name: str
    expression: '_Dimension'
    value: Union[float, int, str]

    def __str__(self):
        return f'cond({self.expression} {self.name} {self.value})'

    def visit(self, visitor, mode, *args, **kwargs):
        return visitor.cond_leaf(mode, self, *args, **kwargs)


@dataclass
class _NodeCondition(_Condition):
    name: str
    lhs: _Condition
    rhs: _Condition

    def __str__(self):
        return f'({self.lhs} {self.name} {self.rhs})'

    def visit(self, visitor, mode, *args, **kwargs):
        return visitor.cond_node(mode, self, *args, **kwargs)


def either(a, b):
    """True of one of the conditions `a` and `b` are true

    Parameters
    ----------
    a: _Condition

    b: _Condition

    Returns
    -------
    returns a `_Condition`
    """
    return _NodeCondition('or', a, b)


def both(a, b):
    """True if both conditions `a` and `b` are true

    Parameters
    ----------
    a: _Condition

    b: _Condition

    Returns
    -------
    returns a `_Condition`
    """
    return _NodeCondition('and', a, b)


def eq(name, value):
    """True is the sampled value of the hyper-parameter `self` is equal to `value`

    Parameters
    ----------
    name: _Dimension
        hyper-parameter expression

    value: Union[float, int, str]

    Returns
    -------
    returns a `_Condition`
    """
    return _LeafCondition('eq', name, value)


def ne(name, value):
    """True is the sampled value of the hyper-parameter `self` is not equal to `value`

    Parameters
    ----------
    name: _Dimension
        hyper-parameter expression

    value: Union[float, int, str]

    Returns
    -------
    returns a `_Condition`
    """
    return _LeafCondition('ne', name, value)


def lt(name, value):
    """True is the sampled value of the hyper-parameter `self` is less than `value`

    Parameters
    ----------
    name: _Dimension
        hyper-parameter expression

    value: Union[float, int, str]

    Returns
    -------
    returns a `_Condition`
    """
    return _LeafCondition('lt', name, value)


def gt(name, value):
    """True is the sampled value of the hyper-parameter `self` is greater than `value`

    Parameters
    ----------
    name: _Dimension
        hyper-parameter expression

    value: Union[float, int, str]

    Returns
    -------
    returns a `_Condition`
    """
    return _LeafCondition('gt', name, value)


def contains(name, value):
    """True is the sampled value of the hyper-parameter `self` is contained by `value`

    Parameters
    ----------
    name: _Dimension
        hyper-parameter expression

    value: List
        List of values

    Returns
    -------
    returns a `_Condition`
    """
    return _LeafCondition('in', name, value)
