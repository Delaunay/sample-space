from typing import Dict, Union, List, Optional
from dataclasses import dataclass
from collections import OrderedDict
from functools import partial

from sspace.conditionals import eq, ne, lt, gt, contains, both, either, _Condition
from sspace.backends import _OrionSpaceBuilder, _ConfigSpaceBuilder, _Serializer


class _Dimension:
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

    def forbid(self, cond):
        """Forbid the value specified by the condition to be taken by the hyper-parameter"""
        # forbid only has AND
        if self.forbidden is not None:
            cond = both(self.forbidden, cond)

        self.forbidden = cond
        return self

    def enable_if(self, cond):
        """Enable the underlying hyper-parameter only if the condition is true"""
        if self.condition is not None:
            raise RuntimeError('Use `either` or `both` to combine constraint!')

        self.condition = cond
        return self


@dataclass
class _Uniform(_Dimension):
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

    def visit(self, visitor):
        return visitor.dim_leaf(
            'uniform', name=self.name, lower=self.a, upper=self.b,
            discrete=self.discrete, log=self.log, quantization=self.quantization)


@dataclass
class _Normal(_Dimension):
    name: str
    loc: Union[int, float]
    scale: Union[int, float]
    discrete: bool = False
    log: bool = False
    quantization: Union[int, float] = None
    space = None
    condition: Optional[_Condition] = None
    forbidden: Optional[_Condition] = None

    def visit(self, visitor):
        return visitor.dim_leaf(
            'normal', name=self.name, loc=self.loc, scale=self.scale,
            discrete=self.discrete, log=self.log, quantization=self.quantization)


@dataclass
class _Categorical(_Dimension):
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

    def visit(self, visitor):
        return visitor.dim_leaf('categorical', name=self.name, options=self.options)


@dataclass
class _Ordinal(_Dimension):
    name: str
    values: List
    space = None
    condition: Optional[_Condition] = None
    forbidden: Optional[_Condition] = None

    def visit(self, visitor):
        return visitor.dim_leaf('ordinal', name=self.name, sequence=list(self.values))


class Space(_Dimension):
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

    def visit(self, visitor):
        """Run the space builder recursively

        Returns
        -------
        returns the created hyper-parameter space
        """
        return visitor.dim_node(self)

    def _factory(self, type, name, *args, **kwargs):
        p = type(name, *args, **kwargs)
        p.space = self
        self.space_tree[name] = p
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
        >>> space.sample()
        >>> [OrderedDict([('a', 1.52)])]

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
        >>> space.sample()
        >>> [OrderedDict([('a', 1.52)])]

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
        >>> space.sample()
        >>> [OrderedDict([('a', 1.52)])]

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
        >>> space.sample()
        >>> [OrderedDict([('a', 1.52)])]

        Returns
        -------
        returns the created hyper-parameter
        """
        return self._factory(_Normal, name, loc, scale, discrete, True, quantization)

    def ordinal(self, name, *values):
        """Add a new ordinal hyper-parameter

        Parameters
        ----------
        name: str
            Name of the hyper-parameter

        values: List
            list of values that are sampled in sequence

        Returns
        -------
        returns the created hyper-parameter
        """
        if len(values) == 1 and isinstance(values[0], list):
            values = values[0]

        return self._factory(_Ordinal, name, values)

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
        >>> subspace = space.subspace('b')
        >>> subspace.normal('a', 1, 2, quantization=0.01)
        >>> OrderedDict([('a', 4.53), ('b.a', 1.8)])

        Returns
        -------
        returns the created hyper-parameter
        """
        return self._factory(Space, name, backend=self.backend)

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

    def instantiate(self, backend=None):
        """Instantiate the underlying sampler for the defined space"""
        if backend is None:
            backend = self.backend

        dispatch = {
            'ConfigSpace': _ConfigSpaceBuilder,
            'Orion': _OrionSpaceBuilder
        }

        backend_type = dispatch.get(backend)
        builder = backend_type()
        self.space_handle = self.visit(builder)
        self.sampler = partial(backend_type.sample, self.space_handle)

        return self.space_handle

    def sample(self, n_samples=1, seed=0):
        """Sample a configuration using the underlying sampler"""
        if self.sampler is None:
            self.instantiate()

        return self.sampler(n_samples, seed)

    def serialize(self):
        return self.visit(_Serializer())

    @staticmethod
    def _parse_cond(cond):
        if cond is None:
            return None

        functions = {
            'eq': eq, 'ne': ne, 'lt': lt, 'gt': gt, 'in': contains,
            'and': both, 'or': either
        }

        assert len(cond) == 1

        function_name, args = cond.popitem()
        fun = functions.get(function_name)

        if function_name in ('and', 'or'):
            lhs, rhs = args
            return fun(Space._parse_cond(lhs), Space._parse_cond(rhs))

        if fun is None:
            raise NotImplementedError(f'function {function_name} is not implemented')

        args['self'] = args.pop('name')
        return fun(**args)

    @staticmethod
    def from_json(data=None, file=None, space=None):
        """
        Parameters
        ----------
        data:
            serialized space (dictionary)

        file:
            load a json file of a serialized space

        space:
            space object to use to recreate the space
        """
        import json

        if file is not None:
            with open(file, 'r') as jfile:
                data = json.load(jfile)

        self = space
        if space is None:
            self = Space()

        for k, kwargs in data.items():
            conditional = kwargs.pop('conditionals', None)
            forbid = kwargs.pop('forbid', None)

            # If k is a function then we can create the hyper param
            if hasattr(self, k):
                hp = getattr(self, k)(**kwargs)

                hp.condition = Space._parse_cond(conditional)
                hp.forbidden = Space._parse_cond(forbid)

            else:  # it has to be a sub space
                subspace = self.subspace(k)
                Space.from_json(kwargs, space=subspace)

        return self
