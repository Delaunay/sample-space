from dataclasses import dataclass
from collections import OrderedDict
from functools import partial
import hashlib
from typing import Dict, Union, List, Optional

from sspace.conditionals import eq, ne, lt, gt, contains, both, either, _Condition
from sspace.backends import _OrionSpaceBuilder, _ConfigSpaceBuilder, _ShortSerializer


class MissingVariable(Exception):
    """Raised when a variable is missing"""
    pass


class Dimension:
    """Base Node of a simple graph structure. All node are leaves except ofr the Space node"""

    def __eq__(self, other):
        return eq(self, other)

    def __ne__(self, other):
        return ne(self, other)

    def __lt__(self, other):
        return lt(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __in__(self, other):
        return contains(self, other)

    def forbid_equal(self, value):
        """Forbid the value to be taken by the hyper-parameter

        Examples
        --------
        >>> from sspace import Space
        >>> space = Space()
        >>> a = space.uniform('a', 1, 2)
        >>> b = space.uniform('b', 1, 2)
        >>> b.forbid_equal(1)
        """
        cond = eq(self, value)

        # forbid only has AND
        if self.forbidden is not None:
            cond = both(self.forbidden, cond)

        self.forbidden = cond
        return

    def forbid_in(self, values):
        """Forbid a set of values to be taken by the hyper-parameter

        Examples
        --------
        >>> from sspace import Space
        >>> space = Space()
        >>> a = space.uniform('a', 1, 2)
        >>> b = space.uniform('b', 1, 2)
        >>> b.forbid_in([1, 2, 3])

        """
        cond = contains(self, values)

        # forbid only has AND
        if self.forbidden is not None:
            cond = both(self.forbidden, cond)

        self.forbidden = cond
        return

    def enable_if(self, cond):
        """Enable the underlying hyper-parameter only if the condition is true

        Examples
        --------
        >>> from sspace import Space
        >>> space = Space()
        >>> a = space.uniform('a', 1, 2)
        >>> b = space.uniform('b', 1, 2)
        >>> b.enable_if(either(gt(a, 2), lt(a, 1)))
        """
        if self.condition is not None:
            raise RuntimeError('Use `either` or `both` to combine constraint!')

        self.condition = cond
        return None

    def visit(self, visitor, *args, **kwargs):
        raise NotImplementedError()


@dataclass
class _Uniform(Dimension):
    name: str
    a: Union[int, float]
    b: Union[int, float]
    discrete: bool = False
    log: bool = False
    quantization: Union[int, float] = None
    space = None
    condition: Optional[_Condition] = None
    forbidden: Optional[_Condition] = None

    @property
    def lower(self):
        return self.a

    @property
    def upper(self):
        return self.b

    def visit(self, visitor, *args, **kwargs):
        return visitor.dim_leaf(
            'uniform', *args, name=self.name, lower=self.a, upper=self.b,
            discrete=self.discrete, log=self.log, quantization=self.quantization, **kwargs)

    def __repr__(self):
        fun = 'uniform'
        if self.log:
            fun = 'loguniform'

        b = f'{fun}({self.name}, upper={self.a}, lower={self.b}, discrete={self.discrete}'
        if self.quantization:
            return b + f', q={self.quantization})'
        return b + ')'


@dataclass
class _Normal(Dimension):
    name: str
    loc: Union[int, float]
    scale: Union[int, float]
    discrete: bool = False
    log: bool = False
    quantization: Union[int, float] = None
    space = None
    condition: Optional[_Condition] = None
    forbidden: Optional[_Condition] = None

    def visit(self, visitor, *args, **kwargs):
        return visitor.dim_leaf(
            'normal', *args, name=self.name, loc=self.loc, scale=self.scale,
            discrete=self.discrete, log=self.log, quantization=self.quantization, **kwargs)

    def __repr__(self):
        fun = 'normal'
        if self.log:
            fun = 'lognormal'

        b = f'{fun}({self.name}, loc={self.loc}, scale={self.scale}, discrete={self.discrete}'
        if self.quantization:
            return b + f', q={self.quantization})'
        return b + ')'


@dataclass
class _Categorical(Dimension):
    name: str
    options: Dict[str, float]
    space = None
    condition: Optional[_Condition] = None
    forbidden: Optional[_Condition] = None

    @property
    def choices(self):
        return list(self.options.keys())

    @property
    def weights(self):
        return list(self.options.values())

    def visit(self, visitor, *args, **kwargs):
        return visitor.dim_leaf('categorical', *args, name=self.name, options=self.options, **kwargs)

    def __repr__(self):
        return f'choices({self.name}, {", ".join([f"{k}={v}" for k, v in self.options.items()])})'


@dataclass
class _Ordinal(Dimension):
    name: str
    values: List
    space = None
    condition: Optional[_Condition] = None
    forbidden: Optional[_Condition] = None

    def visit(self, visitor, *args, **kwargs):
        return visitor.dim_leaf('ordinal', *args, name=self.name, sequence=list(self.values), **kwargs)

    def __repr__(self):
        return f'ordinal({self.name}, {self.values})'


@dataclass
class _Variable(Dimension):
    name: str

    def visit(self, visitor, *args, **kwargs):
        return visitor.dim_leaf('var', *args, name=self.name, **kwargs)

    def __repr__(self):
        return f'var({self.name})'


def compute_identity(sample, size):
    """Compute a unique hash out of a dictionary

    Parameters
    ----------
    sample: dict
        Dictionary to compute the hash from

    size: int
        size of the unique hash
    """
    sample_hash = hashlib.sha256()

    for k, v in sorted(sample.items()):
        sample_hash.update(k.encode('utf8'))

        if isinstance(v, (dict, OrderedDict)):
            sample_hash.update(compute_identity(v, size).encode('utf8'))
        else:
            sample_hash.update(str(v).encode('utf8'))

    return sample_hash.hexdigest()[:size]


class Space(Dimension):
    """Multi Dimension hyper-parameter space

    Arguments
    ---------
    name: Optional[str]
        Name of the hyper-parameter space

    backend: str
        Name of the sampler backend to use (default: ConfigSpace)
        choice between `ConfigSpace` and `Orion`
    """
    def __init__(self, name=None, backend='ConfigSpace'):
        self.name = name
        self.space_tree = OrderedDict()
        self.space = None

        self.condition: Optional[_Condition] = None
        self.forbidden: Optional[_Condition] = None
        self.space_builder = None
        self.space_handle = None
        self.sampler = None
        self.backend = backend
        self.variables = {}
        self._identity = None
        self._identity_size = 16
        self.parent = None

    def visit(self, visitor, *args, **kwargs):
        """Run the space builder recursively

        Returns
        -------
        returns the created hyper-parameter space
        """
        return visitor.dim_node(self, *args, **kwargs)

    def _factory(self, ctor, name, *args, **kwargs):
        namespaces = name.split('.')

        prev = self.space_tree
        parent_space = self
        last_name = namespaces[-1]

        for i, namespace in enumerate(namespaces[:-1]):
            new_space = prev.get(namespace)

            if new_space is None:
                parent_space = parent_space.subspace(namespace)
                prev = parent_space.space_tree

            elif isinstance(new_space, Space):
                parent_space = new_space
                prev = new_space.space_tree

            else:
                last_name = '.'.join(namespaces[i:])
                break

        p = ctor(name, *args, **kwargs)
        p.space = parent_space
        prev[last_name] = p
        return p

    def uniform(self, name, lower, upper, discrete=False, log=False, quantization=None):
        """Add a new normal hyper-parameter

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        lower: Union[float, int]
            lower value

        upper: Union[float, int]
            upper value

        discrete: bool
            is the distribution integer (discrete=True) of float (discrete=False)

        log: bool
            is the distribution log

        quantization: Union[float, int]
            Truncation factor (quantization=0.01 will limit the number of decimals to 2)

        Examples
        --------

        >>> space = Space()
        >>> space.uniform('a', 1, 2, quantization=0.01)
        uniform(a, upper=1, lower=2, discrete=False, q=0.01)
        >>> space.sample()
        [OrderedDict([('a', 1.55)])]

        Returns
        -------
        returns the created hyper-parameter
        """
        return self._factory(_Uniform, name, lower, upper, discrete, log, quantization)

    def loguniform(self, name, lower, upper, discrete=False, quantization=None):
        """Add a new normal hyper-parameter

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        lower: Union[float, int]
            lower value

        upper: Union[float, int]
            upper value

        discrete: bool
            is the distribution integer (discrete=True) of float (discrete=False)

        quantization: Union[float, int]
            Truncation factor (quantization=0.01 will limit the number of decimals to 2)

        Examples
        --------

        >>> space = Space()
        >>> space.loguniform('a', 1, 2, quantization=0.01)
        loguniform(a, upper=1, lower=2, discrete=False, q=0.01)
        >>> space.sample()
        [OrderedDict([('a', 1.46)])]

        Returns
        -------
        returns the created hyper-parameter
        """
        return self._factory(_Uniform, name, lower, upper, discrete, True, quantization)

    def normal(self, name, loc, scale, discrete=False, log=False, quantization=None):
        """Add a new normal hyper-parameter

         Parameters
        ----------
        name: str
            Name of the hyper-parameter

        loc: Union[float, int]
            mean of the distribution

        scale: Union[float, int]
            standard deviation of the distribution

        discrete: bool
            is the distribution integer (discrete=True) of float (discrete=False)

        log: bool
            is the distribution log

        quantization: Union[float, int]
            Truncation factor (quantization=0.01 will limit the number of decimals to 2)

        Examples
        --------

        >>> space = Space()
        >>> space.normal('a', 1, 2, quantization=0.01)
        normal(a, loc=1, scale=2, discrete=False, q=0.01)
        >>> space.sample()
        [OrderedDict([('a', 4.53)])]

        Returns
        -------
        returns the created hyper-parameter
        """
        return self._factory(_Normal, name, loc, scale, discrete, log, quantization)

    def lognormal(self, name, loc, scale, discrete=False, quantization=None):
        """Add a new log-normal hyper-parameter

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        loc: Union[float, int]

        scale: Union[float, int]

        discrete: bool
            is the distribution integer (discrete=True) of float (discrete=False)

        quantization: Union[float, int]
            Truncation factor (quantization=0.01 will limit the number of decimals to 2)

        Examples
        --------

        >>> space = Space()
        >>> space.lognormal('a', 1, 2, quantization=0.01)
        lognormal(a, loc=1, scale=2, discrete=False, q=0.01)
        >>> space.sample()
        [OrderedDict([('a', 92.58)])]

        Returns
        -------
        returns the created hyper-parameter
        """
        return self._factory(_Normal, name, loc, scale, discrete, True, quantization)

    def ordinal(self, name, *values, sequence=None):
        """Add a new ordinal hyper-parameter, ordinals are sampled in-order

        Notes
        -----

        This is particularly useful if you want to specify something like epoch in your search space, and
        make your search space change in function of the epoch.

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        values: List
            list of values that are sampled in sequence

        Examples
        --------

        >>> space = Space()
        >>> space.ordinal('a', 1, 2, 3, 4, 5)
        ordinal(a, (1, 2, 3, 4, 5))
        >>> space.sample()
        [OrderedDict([('a', 5)])]
        >>> space.sample(seed=1)
        [OrderedDict([('a', 4)])]

        Returns
        -------
        returns the created hyper-parameter
        """
        if len(values) == 1 and isinstance(values[0], list):
            values = values[0]

        elif sequence is not None:
            values = sequence

        return self._factory(_Ordinal, name, tuple(list(values)))

    def variable(self, name=None):
        """Add a variable parameter, variables are set by outside actors.
        It is a way for hyper parameter optimizer to set constraint before the sampling

        Examples
        --------

        >>> space = Space()
        >>> space.variable('epoch')
        var(epoch)
        >>> space.sample(seed=1, epoch=1)
        [OrderedDict([('epoch', 1)])]

        """
        if self.parent is not None:
            self.parent.variable(f'{self.name}.{name}')
        else:
            p = _Variable(name=name)
            self.variables[name] = _Variable(name=name)
            return p

    def identity(self, name, size=16):
        """Identity is use to compute a hash to uniquely identify samples.
        For complex research spaces the seed can be used as an identity proxy as the
        probabilities for getting two exact same sample for different seed is low.

        Parameters
        ----------
        name: str
            name of the identity parameter

        size: int
            size of the digest to keep

        Examples
        --------

        >>> space = Space()
        >>> space.uniform('a', 0, 1)
        uniform(a, upper=0, lower=1, discrete=False)
        >>> space.identity('uid')
        >>> space.sample()
        [OrderedDict([('a', 0.5488135039273248), ('uid', 'ac1301101b979371')])]

        """
        self._identity = name
        self._identity_size = size

    def subspace(self, name):
        """Insert a new hyper parameter subspace

        Parameters
        ----------
        name: str
            Name of new the hyper-parameter space

        Examples
        --------

        >>> space = Space()
        >>> space.normal('a', 1, 2, quantization=0.01)
        normal(a, loc=1, scale=2, discrete=False, q=0.01)
        >>> subspace = space.subspace('b')
        >>> subspace.normal('a', 1, 2, quantization=0.01)
        normal(a, loc=1, scale=2, discrete=False, q=0.01)
        >>> space.sample()
        [OrderedDict([('a', 4.53), ('b', OrderedDict([('a', 1.8)]))])]

        Returns
        -------
        returns the created hyper-parameter
        """
        space = self._factory(Space, name, backend=self.backend)
        space.parent = self
        return space

    def categorical(self, name, options=None, **options_w):
        """Add a categorical hyper-parameters, sampled from a set of values

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        options: List
            List of choices that are available

        options_w: Dict[str, float]
            Dictionary with keys as the choices and the values as the weight

        Examples
        --------

        >>> space = Space()
        >>> space.categorical('a', ['v1', 'v2'])
        choices(a, v1=0.5, v2=0.5)
        >>> space.sample()
        [OrderedDict([('a', 'v2')])]

        categorical also accepts custom probability weights

        >>> space = Space()
        >>> space.categorical('a', {'v1': 0.1, 'v2': 0.2, 'v3': 0.7})
        choices(a, v1=0.1, v2=0.2, v3=0.7)
        >>> space.sample()
        [OrderedDict([('a', 'v3')])]

        Returns
        -------
        returns the created hyper-parameter
        """
        if options is not None and isinstance(options, list):
            for v in options:
                options_w[v] = 1 / len(options)

        if options is not None and isinstance(options, dict):
            options_w = options

        return self._factory(_Categorical, name, options_w)

    def choices(self, name, options=None, **options_w):
        """Same as `Space.categorical`"""
        return self.categorical(name, options, **options_w)

    def instantiate(self, backend=None):
        """Instantiate the underlying sampler for the defined space"""
        if backend is None:
            backend = self.backend

        dispatch = {
            'ConfigSpace': _ConfigSpaceBuilder,
            'Orion': _OrionSpaceBuilder
        }

        backend_type = dispatch.get(backend)
        if backend_type is None:
            raise RuntimeError(f'Backend {backend} does not exist pick one from {list(dispatch.keys())}')

        builder = backend_type()
        self.space_handle = self.visit(builder)
        self.sampler = partial(backend_type.sample, self.space_handle)

        return self.space_handle

    def sample(self, n_samples=1, seed=0, **variables):
        """Sample a configuration using the underlying sampler.

        Parameters
        ----------
        n_samples: int
            number of samples to generate

        seed: int
            seed of PRNG

        variables:
            Variable definitions

        Notes
        -----

        Space sampler is entirely deterministic;
        you need to change the seed to generate different samples

        Examples
        --------

        >>> space = Space()
        >>> space.uniform('a', 0, 1)
        uniform(a, upper=0, lower=1, discrete=False)
        >>> space.sample()
        [OrderedDict([('a', 0.5488135039273248)])]
        >>> space.sample()
        [OrderedDict([('a', 0.5488135039273248)])]
        >>> space.sample(seed=1)
        [OrderedDict([('a', 0.417022004702574)])]

        The samples format makes it easy to transform them into a pandas DataFrame if needed

        >>> import pandas as pd
        >>> space = Space()
        >>> space.uniform('a', 0, 1)
        uniform(a, upper=0, lower=1, discrete=False)
        >>> samples = pd.DataFrame(space.sample(10))
        >>> samples
                  a
        0  0.548814
        1  0.715189
        2  0.602763
        3  0.544883
        4  0.423655
        5  0.645894
        6  0.437587
        7  0.891773
        8  0.963663
        9  0.383442
        >>> dict(zip(samples.keys(), samples.values[0]))
        {'a': 0.5488135039273248}

        """
        if self.sampler is None:
            self.instantiate()

        for v in self.variables.keys():
            if v not in variables:
                raise MissingVariable(f'Variable {v} is missing in keyword arguments')

        samples = self.sampler(n_samples, seed)

        for s in samples:
            # identity takes variables & samples into account
            s.update(variables)

            if self._identity is not None:
                s[self._identity] = compute_identity(s, self._identity_size)

        return [self.unflatten(s) for s in samples]

    def unflatten(self, dictionary):
        """Unflatten the a dictionary using the space to know when to unflatten or not"""
        new_dict = OrderedDict()
        for k, v in dictionary.items():
            namespaces = k.split('.')
            last_name = namespaces[-1]

            prev = new_dict
            parent = self

            for i, namespace in enumerate(namespaces[:-1]):
                obj = parent.space_tree.get(namespace)

                if isinstance(obj, Space):
                    parent = obj
                    t = prev.get(namespace, OrderedDict())
                    prev[namespace] = t
                    prev = t
                else:
                    last_name = '.'.join(namespaces[i:])
                    break

            prev[last_name] = v

        return new_dict

    def serialize(self):
        """Serialize a space into a python dictionary/json"""
        return self.visit(_ShortSerializer())

    @staticmethod
    def from_json(file, space=None):
        """Load a serialized space from a json file

        Parameters
        ----------
        file:
            load a json file of a serialized space

        space:
            space object to use to recreate the space
        """
        import json

        with open(file, 'r') as jfile:
            return Space.from_dict(json.load(jfile), space)

    @staticmethod
    def from_dict(data, space=None):
        """Load a serialized space from a python dictionary

        Parameters
        ----------
        data:
            serialized space (dictionary)

        space:
            space object to use to recreate the space
        """
        self = space
        if space is None:
            self = Space()

        return _ShortSerializer.deserialize(data, self)
