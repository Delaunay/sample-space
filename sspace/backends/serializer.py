
class _Serializer:
    def cond_leaf(self, mode, leaf, hyper_parameter, ctx=None):
        name = leaf.expression
        if not isinstance(name, str):
            name = name.name

        cond = {leaf.name: {
            'name': name,
            'value': leaf.value
        }}

        if hyper_parameter is not None:
            hyper_parameter[mode] = cond

        return cond

    def cond_node(self, mode, node, hyper_parameter, ctx=None):
        lhs = node.lhs.visit(self, mode, None, ctx)
        rhs = node.rhs.visit(self, mode, None, ctx)

        hyper_parameter[mode] = {
            node.name: [lhs, rhs]
        }
        return

    def dim_leaf(self, fun_name, **kwargs):
        return {fun_name: kwargs}

    def dim_node(self, node, **kwargs):
        space = {}

        for k, hp_expr in node.space_tree.items():
            new_hp = hp_expr.visit(self, **kwargs)
            space.update(new_hp)

            # new_hp = {hp_name: **kwargs}
            hp_def = list(new_hp.values())[0]

            if hp_expr.condition is not None:
                hp_expr.condition.visit(self, 'conditionals', hp_def)

            if hp_expr.forbidden is not None:
                hp_expr.forbidden.visit(self, 'forbid', hp_def)

        if node.name is None:
            return space
        return {node.name: space}
