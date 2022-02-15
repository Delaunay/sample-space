import_error = None

try:
    import ConfigSpace as cs
    import ConfigSpace.hyperparameters as csh

    from orion.core.io.space_builder import SpaceBuilder, DimensionBuilder
    import orion.algo.space as OrionSpace
    import functools

    from sspace.utils import sort_dict


except ImportError as e:
    import_error = e


def convert_space(space):
    """Convert space dictionary into a ConfigSpace"""
    if import_error:
        raise import_error

    def ignore(self):
        return None

    _convert_table = {
        OrionSpace.Real: _convert_real,
        OrionSpace.Integer: _convert_int,
        OrionSpace.Categorical: _convert_categorical,
        OrionSpace.Fidelity: ignore
    }

    builder = SpaceBuilder()
    orion_space = builder.build(sort_dict(space))

    new_space = cs.ConfigurationSpace()
    for k, v in orion_space.items():
        hp = None

        # --- Dispatch
        fun = _convert_table.get(type(v))

        if fun is not None:
            hp = fun(v)

        else:
            print('Conversion not found for:')
            print('    - ', k, type(v), v)

        # --- Add HP
        if hp is not None:
            new_space.add_hyperparameter(hp)

    return new_space


def _not_implemented(name):
    def exception():
        raise NotImplementedError(f'Prior {name} is unknown')

    return exception


def _convert_real(self):
    a, b = self._args

    def make_uniform():
        return csh.UniformFloatHyperparameter(
            self.name, lower=a, upper=b, default_value=self.default_value, q=None, log=False)

    def make_normal():
        return csh.NormalFloatHyperparameter(
            self.name, mu=a, sigma=b, default_value=self.default_value, q=None, log=False)

    def make_loguniform():
        return csh.UniformFloatHyperparameter(
            self.name, lower=a, upper=b, default_value=self.default_value, q=None, log=True)

    _prior_dispatch = {
        'uniform': make_uniform,
        'norm': make_normal,
        'normal': make_normal,
        'reciprocal': make_loguniform
    }

    return _prior_dispatch.get(self._prior_name, _not_implemented(self._prior_name))()


def _convert_int(self):
    a, b = self._args

    def make_uniform():
        return csh.UniformIntegerHyperparameter(
            self.name, lower=a, upper=b, default_value=self.default_value, q=None, log=False)

    def make_loguniform():
        return csh.UniformIntegerHyperparameter(
            self.name, lower=a, upper=b, default_value=self.default_value, q=None, log=True)

    def make_normal():
        return csh.NormalIntegerHyperparameter(
            self.name, mu=a, sigma=b, default_value=self.default_value, q=None, log=False)


    _prior_dispatch = {
        'uniform': make_uniform,
        'reciprocal': make_loguniform,
        'norm': make_normal,
        'normal': make_normal,
    }

    return _prior_dispatch.get(self._prior_name, _not_implemented(self._prior_name))()


def _convert_categorical(self):
    def make_categorical():
        return csh.CategoricalHyperparameter(self.name, choices=self.categories, weights=self._probs)

    _prior_dispatch = {
        'Distribution': make_categorical,
    }

    return _prior_dispatch.get(self._prior_name, _not_implemented(self._prior_name))()




