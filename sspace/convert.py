import_error = None

try:
    import ConfigSpace as cs
    import ConfigSpace.hyperparameters as csh

    from sspace.orion.builder import SpaceBuilder, DimensionBuilder
    import sspace.orion.space as OrionSpace
    import functools

    from sspace.utils import sort_dict

except ImportError as e:
    import_error = e



_NoneValue = object()


def transform(params):
    """Convert from ConfigSpace params to Orion params"""
    return {k: (v if v is not _NoneValue else None) for k, v in params.items()}


def reverse(params):
    """Convert from Orion params to ConfigSpace params"""
    return {k: (v if v is not None else _NoneValue) for k, v in params.items()}


def build_space(space):
    builder = SpaceBuilder()
    return builder.build(sort_dict(space))


def convert_space(orion_space):
    """Convert space dictionary into a ConfigSpace"""
    if import_error:
        raise import_error

    def ignore(self):
        return None

    _convert_table = {
        "real": _convert_real,
        "integer": _convert_int,
        "categorical": _convert_categorical,
        "fidelity": ignore,
    }

    new_space = cs.ConfigurationSpace()
    for k, v in orion_space.items():
        hp = None

        # --- Dispatch
        fun = _convert_table.get(v.type)

        if fun is not None:
            hp = fun(v)

        else:
            print("Conversion not found for:")
            print("    - ", k, type(v), v)

        # --- Add HP
        if hp is not None:
            new_space.add_hyperparameter(hp)

    return new_space


def _not_implemented(name):
    def exception():
        raise NotImplementedError(f'Prior {name} is unknown')

    return exception


def _convert_real(self):
    a, b = self.interval()

    def make_uniform():
        return csh.UniformFloatHyperparameter(
            self.name,
            lower=a,
            upper=b,
            default_value=self.default_value,
            q=None,
            log=False,
        )

    def make_normal():
        return csh.NormalFloatHyperparameter(
            self.name,
            mu=a,
            sigma=b,
            default_value=self.default_value,
            q=None,
            log=False,
        )

    def make_loguniform():
        return csh.UniformFloatHyperparameter(
            self.name,
            lower=a,
            upper=b,
            default_value=self.default_value,
            q=None,
            log=True,
        )

    _prior_dispatch = {
        'uniform': make_uniform,
        'norm': make_normal,
        'normal': make_normal,
        # TODO: Need to support access to prior object through transformation
        "norm": make_normal,
        'reciprocal': make_loguniform
    }

    return _prior_dispatch.get(self.prior_name, _not_implemented(self.prior_name))()


def _convert_int(self):
    a, b = self.interval()

    def make_uniform():
        return csh.UniformIntegerHyperparameter(
            self.name, lower=a, upper=b, default_value=self.default_value, q=None, log=False)

    def make_loguniform():
        return csh.UniformIntegerHyperparameter(
            self.name, lower=a, upper=b, default_value=self.default_value, q=None, log=True)

    def make_normal():
        return csh.NormalIntegerHyperparameter(
            self.name, mu=a, sigma=b, default_value=self.default_value, q=None, log=False)

    def make_normal():
        return csh.NormalIntegerHyperparameter(
            self.name,
            mu=a,
            sigma=b,
            default_value=self.default_value,
            q=None,
            log=False,
        )

    _prior_dispatch = {
        "uniform": make_uniform,
        "reciprocal": make_loguniform,

        "int_uniform": make_uniform,
        "int_reciprocal": make_loguniform,

        # TODO: Need to support access to prior object through transformation
        "int_norm": make_normal,
        'norm': make_normal,
        'normal': make_normal,
    }

    return _prior_dispatch.get(self.prior_name, _not_implemented(self.prior_name))()


def _convert_categorical(self):
    choices = tuple(v if v is not None else _NoneValue for v in self.interval())

    def make_categorical():
        return csh.CategoricalHyperparameter(
            self.name,
            choices=choices,
            # TODO: Need to support access to probs attribute through transformation
            weights=self._probs if hasattr(self, "_probs") else None,
        )

    return make_categorical()
