# -*- coding: utf-8 -*-
"""
Flatten and unflatten dicts
===========================

Turn deep dictionaries into flat key.subkey versions and vice-versa.

"""


import copy


def flatten(dictionary):
    """Turn all nested dict keys into a {key}.{subkey} format"""

    def _flatten(dictionary):
        if dictionary == {}:
            return dictionary

        key, value = dictionary.popitem()
        if not isinstance(value, dict) or not value:
            new_dictionary = {key: value}
            new_dictionary.update(flatten(dictionary))
            return new_dictionary

        flat_sub_dictionary = flatten(value)
        for flat_sub_key in list(flat_sub_dictionary.keys()):
            flat_key = key + "." + flat_sub_key
            flat_sub_dictionary[flat_key] = flat_sub_dictionary.pop(flat_sub_key)

        new_dictionary = flat_sub_dictionary
        new_dictionary.update(flatten(dictionary))
        return new_dictionary

    return _flatten(copy.deepcopy(dictionary))


def unflatten(dictionary):
    """Turn all keys with format {key}.{subkey} into nested dictionaries"""
    unflattened_dictionary = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        sub_dictionary = unflattened_dictionary
        for part in parts[:-1]:
            if part not in sub_dictionary:
                sub_dictionary[part] = dict()
            sub_dictionary = sub_dictionary[part]
        sub_dictionary[parts[-1]] = value
    return unflattened_dictionary


def float_to_digits_list(number):
    """Convert a float into a list of digits, without conserving exponant"""
    # Get rid of scientific-format exponant
    str_number = str(number)
    str_number = str_number.split("e")[0]

    res = [int(ele) for ele in str_number if ele.isdigit()]

    # Remove trailing 0s in front
    while len(res) > 1 and res[0] == 0:
        res.pop(0)

    # Remove training 0s at end
    while len(res) > 1 and res[-1] == 0:
        res.pop(-1)

    return res


def tuple_to_dict(data, space):
    """Create a `orion.core.worker.trial.Trial` object from `data`.

    Parameters
    ----------
    data: tuple
        A tuple representing a sample point from `space`.

    space: `orion.algo.space.Space`
        Definition of problem's domain.

    Returns
    -------
    A dictionary
    """
    if len(data) != len(space):
        raise ValueError(
            f"Data point is not compatible with search space:\ndata: {data}\nspace: {space}"
        )

    params = {}
    for i, dim in enumerate(space.values()):
        # params.append(dict(name=dim.name, type=dim.type, value=data[i]))
        params[dim.name] = data[i]

    return params
