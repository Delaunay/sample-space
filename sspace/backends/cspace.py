import_error = None

try:
    import ConfigSpace as cs
    import ConfigSpace.hyperparameters as csh

    from sspace.utils import sort_dict

    cond_dispatch_leaves = {
        'eq': cs.EqualsCondition,
        'ne': cs.NotEqualsCondition,
        'lt': cs.LessThanCondition,
        'gt': cs.GreaterThanCondition,
        'in': cs.InCondition
    }

    forbid_dispatch_leaves = {
        'eq': cs.ForbiddenEqualsClause,
        'in': cs.ForbiddenInClause,
    }

    cond_dispatch_nodes = {
        'and': cs.AndConjunction,
        'or': cs.OrConjunction,
    }

    forbid_dispatch_nodes = {
        'and': cs.ForbiddenAndConjunction
    }

    def categorical(name, options):
        return csh.CategoricalHyperparameter(
            name,
            choices=list(options.keys()),
            weights=list(options.values()))


    def uniform(quantization, discrete=False, **kwargs):
        if discrete:
            return csh.UniformIntegerHyperparameter(q=quantization, **kwargs)
        return csh.UniformFloatHyperparameter(q=quantization, **kwargs)

    def normal(quantization, loc, scale, discrete=False, **kwargs):
        if discrete:
            return csh.NormalIntegerHyperparameter(mu=loc, sigma=scale, q=quantization, **kwargs)
        return csh.NormalFloatHyperparameter(mu=loc, sigma=scale, q=quantization, **kwargs)

    dim_leaves = {
        'uniform': uniform,
        'normal': normal,
        'categorical': categorical,
        'ordinal': csh.OrdinalHyperparameter
    }

except ImportError as e:
    import_error = e


class _ConfigSpaceBuilder:
    def __init__(self):
        if import_error:
            raise import_error

    def cond_leaf(self, mode, leaf, hyper_parameter, ctx=None):
        fun = None

        if mode == 'conditionals':
            fun = cond_dispatch_leaves.get(leaf.name)
        elif mode == 'forbid':
            fun = forbid_dispatch_leaves.get(leaf.name)

        if fun is None:
            raise NotImplementedError(f'{leaf.name} is missing')

        if mode == 'conditionals':
            name = leaf.expression
            if not isinstance(name, str):
                name = name.name

            expr = ctx.get(name)
            return fun(hyper_parameter, expr, leaf.value)

        return fun(hyper_parameter, leaf.value)

    def cond_node(self, mode, node, hyper_parameter, ctx=None):
        fun = None

        if mode == 'conditionals':
            fun = cond_dispatch_nodes.get(node.name)

        elif mode == 'forbid':
            fun = forbid_dispatch_nodes.get(node.name)

        if fun is None:
            raise NotImplementedError(f'{node.name} is missing')

        lhs = node.lhs.visit(self, mode, hyper_parameter, ctx)
        rhs = node.rhs.visit(self, mode, hyper_parameter, ctx)

        return fun(lhs, rhs)

    def dim_leaf(self, fun_name, **kwargs):
        fun = dim_leaves.get(fun_name)

        if fun is None:
            raise NotImplementedError(f'{fun_name} is missing')

        return fun(**kwargs)

    def dim_node(self, node, **kwargs):
        space = cs.ConfigurationSpace()
        ctx = {}

        for k, hp_expr in node.space_tree.items():
            new_hp = hp_expr.visit(self, **kwargs)

            if isinstance(new_hp, cs.ConfigurationSpace):
                space.add_configuration_space(
                    prefix=hp_expr.name,
                    delimiter='.',
                    configuration_space=new_hp)
            else:
                space.add_hyperparameter(new_hp)

            ctx[k] = new_hp

            if hp_expr.condition is not None:
                cond = hp_expr.condition.visit(self, 'conditionals', new_hp, ctx)
                space.add_condition(cond)

            if hp_expr.forbidden is not None:
                forbid = hp_expr.forbidden.visit(self, 'forbid', new_hp)
                space.add_forbidden_clause(forbid)

        return space

    @staticmethod
    def sample(handle, n_samples, seed):
        handle.seed(seed)
        samples = handle.sample_configuration(n_samples)

        if isinstance(samples, list):
            return [sort_dict(c.get_dictionary()) for c in samples]

        return [sort_dict(samples.get_dictionary())]
