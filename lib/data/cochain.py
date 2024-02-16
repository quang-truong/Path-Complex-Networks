"""
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>
Copyright (c) 2021 The CWN Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import torch
import logging
import copy

from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.typing import Adj

from typing import List


class Cochain(object):
    """
    Class representing a cochain on k-dim cells (i.e. vector-valued signals on k-dim cells).

    Args:
        dim: dim of the cells in the cochain
        x: feature matrix, shape [num_cells, num_features]; may not be available
        upper_index: upper adjacency, matrix, shape [2, num_upper_connections];
            may not be available, e.g. when `dim` is the top level dim of a complex
        lower_index: lower adjacency, matrix, shape [2, num_lower_connections];
            may not be available, e.g. when `dim` is 0
        shared_boundaries: a tensor of shape (num_lower_adjacencies,) specifying the indices of
            the shared boundary for each lower adjacency
        shared_coboundaries: a tensor of shape (num_upper_adjacencies,) specifying the indices of
            the shared coboundary for each upper adjacency
        boundary_index: boundary adjacency, matrix, shape [2, num_boundaries_connections];
            may not be available, e.g. when `dim` is 0. First row correspond to (k-1)-cochain indices, 
            second row correspond to k-chain indices.
        upper_orient: a tensor of shape (num_upper_adjacencies,) specifying the relative
            orientation (+-1) with respect to the cells from upper_index
        lower_orient: a tensor of shape (num_lower_adjacencies,) specifying the relative
            orientation (+-1) with respect to the cells from lower_index
        y: labels over cells in the cochain, shape [num_cells,]
    """
    def __init__(self, dim: int, x: Tensor = None, upper_index: Adj = None, lower_index: Adj = None,
                 shared_boundaries: Tensor = None, shared_coboundaries: Tensor = None, mapping: Tensor = None,
                 boundary_index: Adj = None, upper_orient=None, lower_orient=None, y=None, **kwargs):
        if dim == 0:
            assert lower_index is None
            assert shared_boundaries is None
            assert boundary_index is None

        # Note, everything that is not of form __smth__ is made None during batching
        # So dim must be stored like this.
        self.__dim__ = dim
        # TODO: check default for x
        self.__x = x
        self.upper_index = upper_index                              # upper adjacencies (neighbors sharing the same upper complex)
        self.lower_index = lower_index                              # lower adjacencies (neighbors sharing the same lower complex)
        self.boundary_index = boundary_index                        # There is no coboundary_index since co_boundary index is the boundary_index with swapped rows
        self.y = y                                          
        self.shared_boundaries = shared_boundaries                  # the boundary that establish connections in lower_index 
        self.shared_coboundaries = shared_coboundaries              # the coboundary that establish connection in upper_index
        self.upper_orient = upper_orient
        self.lower_orient = lower_orient
        self.__oriented__ = False
        self.__hodge_laplacian__ = None                             # Consider remove
        # TODO: Figure out what to do with mapping.
        self.__mapping = mapping                                    # Look like no use? Consider to remove later
        for key, item in kwargs.items():
            if key == 'num_cells':
                self.__num_cells__ = item
            elif key == 'num_cells_down':
                self.num_cells_down = item
            elif key == 'num_cells_up':
                self.num_cells_up = item
            else:
                self[key] = item

    @property
    def dim(self):
        """Returns the dimension of the cells in this cochain.

        This field should not have a setter. The dimension of a cochain cannot be changed.
        """
        return self.__dim__
    
    @property
    def x(self):
        """Returns the vector values (features) associated with the cells."""
        return self.__x

    @x.setter
    def x(self, new_x):
        """Sets the vector values (features) associated with the cells."""
        if new_x is None:
            logging.warning("Cochain features were set to None. ")
        else:
            assert self.num_cells == len(new_x)
        self.__x = new_x

    @property
    def keys(self):
        """Returns all names of cochain attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]               # get all cochain attributes and methods
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']                # get all cochain attributes and methods that are not __<attribute>__
        return keys

    def __getitem__(self, key):                                                             # Needed for self[key]
        """Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):                                                      # Needed for self[key] = item
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    def __contains__(self, key):
        """Returns :obj:`True`, if the attribute :obj:`key` is present in the data."""
        return key in self.keys

    def __cat_dim__(self, key, value):
        """
        Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.
        """
        if key in ['upper_index', 'lower_index', 'shared_boundaries',
                   'shared_coboundaries', 'boundary_index']:
            return -1           # last dimension since _index matrices have shape (2,n), and _boundaries matrices depend on _index matrices
        # by default, concatenate sparse matrices diagonally.
        elif isinstance(value, SparseTensor):
            return (0, 1)
        return 0

    def __inc__(self, key, value):
        """
        Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.
        """
        # TODO: value is not used in this method. Can it be removed?
        if key in ['upper_index', 'lower_index']:
            inc = self.num_cells
        elif key == 'shared_boundaries':
            inc = self.num_cells_down
        elif key == 'shared_coboundaries':
            inc = self.num_cells_up
        elif key == 'boundary_index':
            boundary_inc = self.num_cells_down if self.num_cells_down is not None else 0
            cell_inc = self.num_cells if self.num_cells is not None else 0
            inc = [[boundary_inc], [cell_inc]]
        else:
            inc = 0
        if inc is None:
            inc = 0

        return inc
    
    def __call__(self, *keys):
        """
        Iterates over all attributes :obj:`*keys` in the cochain, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes.
        """
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    @property
    def num_cells(self):                                                    # num_cells is initialized here: number of k-cells
        """Returns the number of cells in the cochain."""
        if hasattr(self, '__num_cells__'):
            return self.__num_cells__
        if self.x is not None:
            return self.x.size(self.__cat_dim__('x', self.x))               # __cat_dim__('x') return 0, thus x.size(0) = N (N nodes)
        # case when there is no node feature
        if self.boundary_index is not None:                                 # boundary_index has shape (2, num_boundaries_connection) where 1st row is boundary of k-chain, 2nd row is k-chain
            return int(self.boundary_index[1,:].max()) + 1                  # we counting number of k-chain
        assert self.upper_index is None and self.lower_index is None
        return None

    @num_cells.setter
    def num_cells(self, num_cells):
        """Sets the number of cells in the cochain."""
        # TODO: Add more checks here
        self.__num_cells__ = num_cells

    @property
    def num_cells_up(self):                                 # num_cells_up initialized here: number of (k+1)-cells
        """Returns the number of cells in the higher-dimensional cochain of co-dimension 1."""
        if hasattr(self, '__num_cells_up__'):
            return self.__num_cells_up__
        elif self.shared_coboundaries is not None:
            assert self.upper_index is not None
            return int(self.shared_coboundaries.max()) + 1  # count number of (k+1)-cells
        assert self.upper_index is None
        return 0

    @num_cells_up.setter
    def num_cells_up(self, num_cells_up):
        """Sets the number of cells in the higher-dimensional cochain of co-dimension 1."""
        # TODO: Add more checks here
        self.__num_cells_up__ = num_cells_up

    @property
    def num_cells_down(self):                               # num_cells_down initialized here
        """Returns the number of cells in the lower-dimensional cochain of co-dimension 1."""
        if self.dim == 0:
            return None
        if hasattr(self, '__num_cells_down__'):
            return self.__num_cells_down__
        if self.lower_index is None:
            return 0
        raise ValueError('Cannot infer the number of cells in the cochain below.')          # because there may exists isolated nodes.

    @num_cells_down.setter
    def num_cells_down(self, num_cells_down):
        """Sets the number of cells in the lower-dimensional cochain of co-dimension 1."""
        # TODO: Add more checks here
        self.__num_cells_down__ = num_cells_down
        
    @property
    def num_features(self):
        """Returns the number of features per cell in the cochain."""
        if self.x is None:
            return 0
        return 1 if self.x.dim() == 1 else self.x.size(1)

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, SparseTensor):
            # Not all apply methods are supported for `SparseTensor`, e.g.,
            # `contiguous()`. We can get around it by capturing the exception.
            try:
                return func(item)
            except AttributeError:
                return item
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        """
            Applies the function :obj:`func` to all tensor attributes
            :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
            all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        """
            Ensures a contiguous memory layout for all attributes :obj:`*keys`.
            If :obj:`*keys` is not given, all present attributes are ensured to
            have a contiguous memory layout.
        """
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        """
            Performs tensor dtype and/or device conversion to all attributes
            :obj:`*keys`.
            If :obj:`*keys` is not given, the conversion is applied to all present
            attributes.
        """
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def clone(self):
        return self.__class__.from_dict({
            k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
            for k, v in self.__dict__.items()
        })
    
    @property
    def mapping(self):
        return self.__mapping
    

class CochainBatch(Cochain):
    """A datastructure for storing a batch of cochains.

    Similarly to PyTorch Geometric, the batched cochain consists of a big cochain formed of multiple
    independent cochains on sets of disconnected cells.
    """

    def __init__(self, dim, batch=None, ptr=None, **kwargs):
        '''
            dim: dimension of the Cochain in the batch.
            batch: Unclear why (batch is not used)
            
        '''
        super(CochainBatch, self).__init__(dim, **kwargs)

        for key, item in kwargs.items():
            if key == 'num_cells':
                self.__num_cells__ = item
            else:
                self[key] = item

        self.batch = batch
        self.ptr = ptr
        self.__data_class__ = Cochain
        self.__slices__ = None                                  # __slices__ are needed for class to_cochain_list(), which is yet to be defined. kinda keep track of index to slice
        self.__cumsum__ = None                                  
        self.__cat_dims__ = None
        self.__num_cells_list__ = None
        self.__num_cells_down_list__ = None
        self.__num_cells_up_list__ = None
        self.__num_cochains__ = None

    @classmethod
    def from_cochain_list(cls, data_list, follow_batch=[]):             # Factory method (to create many Cochain by calling this function)
        """
            Constructs a batch object from a python list holding
            :class:`Cochain` objects.
            The assignment vector :obj:`batch` is created on the fly.
            Additionally, creates assignment batch vectors for each key in
            :obj:`follow_batch`.
        """
        keys = list(set.union(*[set(data.keys) for data in data_list]))         # get all keys, which are not __<attribute>__, from attributes and methods 
        assert 'batch' not in keys and 'ptr' not in keys                        # make sure there is no batch and ptr in attributes

        batch = cls(data_list[0].dim)                   # so that when CochainBatch.from_cochain_list() is called, batch will be initialized to CochainBatch() with the dimension of the first Cochain in data_list
        for key in data_list[0].__dict__.keys():        # strip off variables that are not  in form __<attribute>__, __<attribute>, or <attribute>__.
            if key[:2] != '__' and key[-2:] != '__':    # hence self.batch and self.ptr is already set to None
                batch[key] = None

        batch.__num_cochains__ = len(data_list)         # __num_cochain__ equal to number of Cochain in data_list
        batch.__data_class__ = data_list[0].__class__   # __data_class__ is Cochain
        for key in keys + ['batch']:                    # empty batch[key] (clear all attributes and methods for every key which is not __<attribute>__)
            batch[key] = []                             # batch.batch = [] ?
        batch['ptr'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}                           # record concat dimension of key
        num_cells_list = []
        num_cells_up_list = []
        num_cells_down_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                if item is not None:
                    # Increase values by `cumsum` value.
                    cum = cumsum[key][-1]               # cumsum[key] is a list, thus retrieving last element (=0 on 1st iter.)
                    if isinstance(item, Tensor) and item.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            item = item + cum
                    elif isinstance(item, SparseTensor):
                        value = item.storage.value()
                        if value is not None and value.dtype != torch.bool:
                            if not isinstance(cum, int) or cum != 0:
                                value = value + cum
                            item = item.set_value(value, layout='coo')
                    elif isinstance(item, (int, float)):
                        item = item + cum

                    # Treat 0-dimensional tensors as 1-dimensional.
                    if isinstance(item, Tensor) and item.dim() == 0:
                        item = item.unsqueeze(0)

                    batch[key].append(item)

                    # Gather the size of the `cat` dimension.
                    size = 1
                    cat_dim = data.__cat_dim__(key, data[key])
                    cat_dims[key] = cat_dim
                    if isinstance(item, Tensor):
                        size = item.size(cat_dim)
                        device = item.device
                    elif isinstance(item, SparseTensor):
                        size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                        device = item.device
                    
                    # TODO: do we really need slices, and, are we managing them correctly?
                    slices[key].append(size + slices[key][-1])          # slice[key] will be appended new value of variable 'size' + previous value of size
                    
                    if key in follow_batch:
                        if isinstance(size, Tensor):
                            for j, size in enumerate(size.tolist()):
                                tmp = f'{key}_{j}_batch'
                                batch[tmp] = [] if i == 0 else batch[tmp]
                                batch[tmp].append(
                                    torch.full((size, ), i, dtype=torch.long,
                                               device=device))
                        else:
                            tmp = f'{key}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))

                inc = data.__inc__(key, item)                           # for example, upper_index inc will the the number of upper_connection. thus the next upper_index from the next Cochain will be increased by the cumsum
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])
            # Done with key here --------------------------
            # Now, consider num_cells per Cochain in Batch
            if hasattr(data, '__num_cells__'):
                num_cells_list.append(data.__num_cells__)
            else:
                num_cells_list.append(None)

            if hasattr(data, '__num_cells_up__'):
                num_cells_up_list.append(data.__num_cells_up__)
            else:
                num_cells_up_list.append(None)

            if hasattr(data, '__num_cells_down__'):
                num_cells_down_list.append(data.__num_cells_down__)
            else:
                num_cells_down_list.append(None)

            num_cells = data.num_cells
            if num_cells is not None:
                item = torch.full((num_cells, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)                        # append (i, i, ...., i) to batch['batch'] (which is initialized with [])
                batch.ptr.append(batch.ptr[-1] + num_cells)     # get new batch['ptr'] value

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]    # initial slice value is set to 0

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_cells_list__ = num_cells_list
        batch.__num_cells_up_list__ = num_cells_up_list
        batch.__num_cells_down_list__ = num_cells_down_list

        ref_data = data_list[0]         # referenced data
        for key in batch.keys:          # concat batch[key] (Tensor, SparseTensor, and Tuple only)
            items = batch[key]
            item = items[0]             # referenced item
            if isinstance(item, Tensor):            # cat 0
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):    # cat (0,1)
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        return batch.contiguous()

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return super(CochainBatch, self).__getitem__(idx)
        elif isinstance(idx, int):
            # TODO: is the 'get_example' method needed for now?
            #return self.get_example(idx)
            raise NotImplementedError
        else:
            # TODO: is the 'index_select' method needed for now?
            # return self.index_select(idx)
            raise NotImplementedError

    def to_cochain_list(self) -> List[Cochain]:
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        # TODO: is the 'to_cochain_list' method needed for now?
        #return [self.get_example(i) for i in range(self.num_cochains)]
        raise NotImplementedError


    @property
    def num_cochains(self) -> int:
        """Returns the number of cochains in the batch."""
        if self.__num_cochains__ is not None:
            return self.__num_cochains__
        return self.ptr.numel() + 1