from .cspace import _ConfigSpaceBuilder
from .ospace import _OrionSpaceBuilder
from .serializer import _Serializer


def orion_space(space):
    """Build an Orion space"""
    builder = _OrionSpaceBuilder()
    return space.visit(builder)


def config_space(space):
    """Build a ConfigSpace space"""
    builder = _ConfigSpaceBuilder()
    return space.visit(builder)
