import_error = None

try:
    import functools

    from orion.core.io.space_builder import SpaceBuilder, DimensionBuilder
    import orion.algo.space as OrionSpace

    from sspace.utils import sort_dict

except ImportError as e:
    import_error = e


class _OrionSpaceBuilder:
    def __init__(self):
        if import_error:
            raise import_error

        self.builder = DimensionBuilder()
        self.dim_leaves = {
            'uniform': self.uniform,
            'normal': self.normal,
            'categorical': self.choices,
            'ordinal': self.ordinal
        }

    def uniform(self, name, lower, upper, log, discrete, quantization=None):
        self.builder.name = name

        if quantization is not None:
            print('Orion does not support quantization')

        if log:
            return self.builder.loguniform(lower, upper, discrete=discrete)

        return self.builder.uniform(lower, upper, discrete=discrete)

    def normal(self, name, loc, scale, log, discrete, quantization=None):
        self.builder.name = name

        if quantization is not None:
            print('Orion does not support quantization')

        if log:
            raise NotImplementedError('Orion does not provide LogNormal')

        return self.builder.normal(loc, scale, discrete=discrete)

    def choices(self, name, options):
        self.builder.name = name
        return self.builder.choices(options)

    def ordinal(self, name, *args):
        raise NotImplementedError('Orion does not provide Ordinal dim')

    def cond_leaf(self, mode, leaf, hyper_parameter, ctx=None):
        raise NotImplementedError('Orion does not support conditionals')

    def cond_node(self, mode, node, hyper_parameter, ctx=None):
        raise NotImplementedError('Orion does not support conditionals')

    def dim_leaf(self, fun_name, **kwargs):
        fun = self.dim_leaves.get(fun_name)

        if fun is None:
            raise NotImplementedError(f'{fun_name} is missing')

        return fun(**kwargs)

    def dim_node(self, node, **kwargs):
        space = OrionSpace.Space()

        for k, dim_expr in node.space_tree.items():
            dim = dim_expr.visit(self)
            space.register(dim)

            if dim_expr.condition is not None:
                print('Orion does not support conditionals')

            if dim_expr.forbidden is not None:
                print('Orion does not support conditionals')

        return space

    @staticmethod
    def _to_dictionary(handle, val):
        return sort_dict({k: v for k, v in zip(handle.keys(), val)})

    @staticmethod
    def sample(handle, n_samples, seed):
        return [_OrionSpaceBuilder._to_dictionary(handle, p) for p in handle.sample(n_samples, seed)]
