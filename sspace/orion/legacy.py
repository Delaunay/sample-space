from sspace.orion.utils import unflatten, flatten


class Param:
    def __init__(self, name, value, type) -> None:
        self.name = name
        self.value = value
        self.type = type


class Trial:
    def __init__(self, params) -> None:
        self._params = [Param(**param) for param in params]

    @property
    def params(self):
        return unflatten({param.name: param.value for param in self._params})


class format_trials:

    @staticmethod
    def tuple_to_trial(data, space, status="new"):
        """Create a `orion.core.worker.trial.Trial` object from `data`.

        Parameters
        ----------
        data: tuple
            A tuple representing a sample point from `space`.
        space: `orion.algo.space.Space`
            Definition of problem's domain.

        Returns
        -------
        A trial object `orion.core.worker.trial.Trial`.
        """
        if len(data) != len(space):
            raise ValueError(
                f"Data point is not compatible with search space:\ndata: {data}\nspace: {space}"
            )

        params = []
        for i, dim in enumerate(space.values()):
            params.append(dict(name=dim.name, type=dim.type, value=data[i]))

        return Trial(params=params)

    @staticmethod
    def trial_to_tuple(trial, space):
        """Extract a parameter tuple from a `orion.core.worker.trial.Trial`.

        The order within the tuple is dictated by the defined
        `orion.algo.space.Space` object.
        """
        params = flatten(trial.params)
        trial_keys = set(params.keys())
        space_keys = set(space.keys())
        if trial_keys != space_keys:
            raise ValueError(
                """"
The trial {} has wrong params:
Trial params: {}
Space dims: {}""".format(
                    trial.id, sorted(trial_keys), sorted(space_keys)
                )
            )
        return tuple(params[name] for name in space.keys())
