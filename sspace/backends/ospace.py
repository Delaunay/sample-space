import functools

from sspace.orion.builder import SpaceBuilder, DimensionBuilder
import sspace.orion.space as OrionSpace

from sspace.utils import sort_dict


class _OrionSpaceBuilder:
    def __init__(self):
        self.builder = DimensionBuilder()
        self.dim_leaves = {
            'uniform': self.uniform,
            'normal': self.normal,
            'categorical': self.choices,
            'ordinal': self.ordinal
        }

    def uniform(self, name, lower, upper, log, discrete, quantization=None):
        self.builder.name = name

        precision = None
        if quantization is not None:
            precision = int(1/quantization)

        if log:
            return self.builder.loguniform(lower, upper, discrete=discrete, precision=precision)

        return self.builder.uniform(lower, upper, discrete=discrete, precision=precision)

    def normal(self, name, loc, scale, discrete, log, quantization=None):
        self.builder.name = name

        precision = None
        if quantization is not None:
            precision = int(1/quantization)

        if log is True:
            raise NotImplementedError('Orion does not provide LogNormal')

        return self.builder.normal(loc, scale, discrete=discrete, precision=precision)

    def choices(self, name, options):
        self.builder.name = name
        return self.builder.choices(options)

    def ordinal(self, name, *args, **kwargs):
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

            if isinstance(dim, OrionSpace.Space):
                for sub_k, sub_dim in dim.items():
                    sub_dim.name = f'{k}.{sub_k}'
                    space.register(sub_dim)
            else:
                space.register(dim)

            if dim_expr.condition is not None:
                print('Orion does not support conditionals')

            if dim_expr.forbidden is not None:
                print('Orion does not support conditionals')

        return space

    @staticmethod
    def _to_dictionary(trial):
        return sort_dict({param.name: param.value for param in trial._params})

    @staticmethod
    def sample(handle, n_samples, seed):
        return [_OrionSpaceBuilder._to_dictionary(t) for t in handle.sample(n_samples, seed)]
