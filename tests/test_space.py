from sspace import Space, either, eq
import json


def make_space():
    space = Space()
    optim = space.categorical('optimizer', ['sgd', 'adam'])
    sgd_lr = space.loguniform('optimizer.lr', 1, 2, quantization=0.01)
    sgd_lr.enable_if(either(eq(optim, 'adam'), eq(optim, 'sgd')))
    sgd_lr.forbid(eq(sgd_lr, 1))
    return space


def test_space_builder_config():
    space = make_space()
    space.config_space()
    space.sample(2)
    json.dumps(space.serialize(), indent=2)


def test_space_builder_orion():
    space = make_space()
    space.orion_space()
    space.sample(2)
    json.dumps(space.serialize(), indent=2)


def test_serialization_is_same():
    import copy

    space = make_space()
    serialized = space.serialize()

    new_space = Space.from_json(copy.deepcopy(serialized))
    new_serialized = new_space.serialize()

    assert serialized == new_serialized


if __name__ == '__main__':
    test_space_builder_config()
    test_space_builder_orion()
    test_serialization_is_same()

