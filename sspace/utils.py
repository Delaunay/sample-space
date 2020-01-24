from collections import OrderedDict


def sort_dict(space):
    if isinstance(space, (dict, OrderedDict)):
        space = list(space.items())

    space.sort(key=lambda i: i[0])
    return OrderedDict(space)
