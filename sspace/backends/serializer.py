from sspace.conditionals import eq, ne, lt, gt, contains, both, either

_functions = {
    'eq': eq, 'ne': ne, 'lt': lt, 'gt': gt, 'in': contains,
    'and': both, 'or': either, 'either': either, 'both': both
}


class _ShortSerializer:
    shortener_small = {
        #    name    log
        ('uniform', True): 'loguniform',
        ('uniform', False): 'uniform',
        ('normal', True): 'lognormal',
        ('normal', False): 'normal',
    }

    rename_node = {
        'or': 'either',
        'and': 'both'
    }

    def cond_leaf(self, mode, leaf, hyper_parameter, ctx=None):
        name = leaf.expression
        if not isinstance(name, str):
            name = name.name

        value = leaf.value
        if isinstance(value, str):
            value = f"'{value}'"

        name = f"'{name}'"
        return f'{leaf.name}(name={name}, value={value})'

    def cond_node(self, mode, node, hyper_parameter, ctx=None):
        lhs = node.lhs.visit(self, mode, None, ctx)
        rhs = node.rhs.visit(self, mode, None, ctx)

        fun_name = _ShortSerializer.rename_node.get(node.name, node.name)
        return f'{fun_name}({lhs}, {rhs})'

    def _short_name_1(self, fun_name, options):
        """Remove one argument for a slightly longer function name"""
        key = (fun_name, options.pop('log', False))
        new_fun = _ShortSerializer.shortener_small.get(key)

        if new_fun is not None:
            return new_fun

        return fun_name

    # def _short_name_2(self, fun_name, options):
    #     """Remove two arguments for a slightly longer function name"""
    #     key = (fun_name, options.pop('discrete', False), options.pop('log', False))
    #     new_fun = _ShortSerializer.shortener_long.get(key)
    #
    #     if new_fun is not None:
    #         return new_fun
    #
    #     return fun_name

    def _simplify(self, kwargs):
        kwargs.pop('name', None)

        # remove default valued arguments
        discrete = kwargs.get('discrete')
        if discrete is not None and discrete is False:
            kwargs.pop('discrete')

        log = kwargs.get('log')
        if log is not None and log is False:
            kwargs.pop('log')

        q = kwargs.get('quantization')
        if q is None:
            kwargs.pop('quantization', None)

    def dim_leaf(self, fun_name, **kwargs):
        fun_name = self._short_name_1(fun_name, kwargs)

        # Name is the dictionary key
        self._simplify(kwargs)

        args = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
        return f'{fun_name}({args})'

    def dim_node(self, node):
        space = {}

        if node._identity is not None:
            name = node._identity
            size = node._identity_size
            space[name] = f'identity(size={size})'

        if node.parent is None:
            for k, v in node.variables.items():
                space[k] = v.visit(self)

        for k, hp_expr in node.space_tree.items():
            kwargs = {}

            if hp_expr.condition is not None:
                kwargs['condition'] = hp_expr.condition.visit(self, 'conditionals', None)

            if hp_expr.forbidden is not None:
                kwargs['forbid'] = hp_expr.forbidden.visit(self, 'forbid', None)

            new_hp = hp_expr.visit(self, **kwargs)
            space[k] = new_hp

        return space

    @staticmethod
    def make_env(space):
        def make_factory(fun_name):
            def factory(name, *args, condition=None, forbid=None, **kwargs):
                p = getattr(space, fun_name)(name, *args, **kwargs)
                p.condition = condition
                p.forbidden = forbid
                return p
            return factory

        def identity(name, *args, **kwargs):
            return space.identity(name, *args)

        def fidelity(name, *args, **kwargs):
            print('fidelity is not supported; it is converted into a variable', args, kwargs)
            return space.variable(name)

        env = {
            'loguniform': make_factory('loguniform'),
            'uniform': make_factory('uniform'),
            'lognormal': make_factory('lognormal'),
            'normal': make_factory('normal'),
            'gaussian': make_factory('normal'),
            'categorical': make_factory('categorical'),
            'ordinal': make_factory('ordinal'),
            'choices': make_factory('categorical'),
            'fidelity': fidelity,
            'var': make_factory('variable'),
            'identity': identity
        }
        env.update(_functions)
        return env

    @staticmethod
    def parse_function_call(fun_call, name, space):
        idx = fun_call.find('(')
        fun_name = fun_call[:idx]

        fun_call = fun_call.replace(f'{fun_name}(', f'{fun_name}(\'{name}\', ')
        return eval(fun_call, _ShortSerializer.make_env(space))

    @staticmethod
    def deserialize(data, space):
        for k, fun_call in data.items():

            # Python code
            if isinstance(fun_call, str):
                _ShortSerializer.parse_function_call(fun_call, k, space)

            else:   # Must be a space
                subspace = space.subspace(k)
                _ShortSerializer.deserialize(fun_call, space=subspace)

        return space


