import pytest

from sspace import Space, either, eq
import json

backends = ['ConfigSpace', 'Orion']


def make_space(backend='ConfigSapce'):
    space = Space(backend=backend)
    optim = space.categorical('optimizer', ['sgd', 'adam'])
    sgd_lr = space.loguniform('optimizer.lr', 1, 2, quantization=0.01)
    sgd_lr.enable_if(either(eq(optim, 'adam'), eq(optim, 'sgd')))
    sgd_lr.forbid(eq(sgd_lr, 1))
    return space


@pytest.mark.parametrize('backend', backends)
def test_space_explicit(backend):
    space = make_space()
    space.instantiate(backend)
    space.sample(2)
    json.dumps(space.serialize(), indent=2)


@pytest.mark.parametrize('backend', backends)
def test_space_implicit(backend):
    space = make_space(backend)
    space.sample(2)
    json.dumps(space.serialize(), indent=2)


@pytest.mark.parametrize('backend', backends)
def test_serialization_is_same(backend):
    import copy

    space = make_space(backend)
    serialized = space.serialize()

    new_space = Space.from_json(copy.deepcopy(serialized))
    new_serialized = new_space.serialize()
    assert serialized == new_serialized


@pytest.mark.parametrize('backend', backends)
def test_subspace(backend):
    space = Space(backend=backend)

    space.normal('a', 1, 2, quantization=0.01)
    subspace = space.subspace('b')
    subspace.normal('a', 1, 2, quantization=0.01)

    print(space.sample())


if __name__ == '__main__':
    for b in backends:
        print(b)
        test_space_explicit(b)
        test_space_implicit(b)
        test_serialization_is_same(b)
        test_subspace(b)
