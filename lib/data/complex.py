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
from torch import Tensor
from typing import List

from .cochain import Cochain, CochainBatch
from lib.message_passing.cochain_mp_params import CochainMessagePassingParams

class Complex(object):          
    """Class representing a cochain complex or an attributed cellular complex.

    Args:
        cochains: A list of cochains forming the cochain complex
        y: A tensor of shape (1,) containing a label for the complex for complex-level tasks.
        dimension: The dimension of the complex.
    """
    def __init__(self, *cochains: Cochain, y: torch.Tensor = None, dimension: int = None): # TODO: Extend to k-dim
        if len(cochains) == 0:
            raise ValueError('At least one cochain is required.')
        if dimension is None:
            dimension = len(cochains) - 1
        if len(cochains) < dimension + 1:
            raise ValueError(f'Not enough cochains passed, '
                             f'expected {dimension + 1}, received {len(cochains)}')

        self.dimension = dimension
        self.cochains = {i: cochains[i] for i in range(dimension + 1)}
        self.nodes = cochains[0]                                            # redundant. only use for testing. TODO: remove
        self.edges = cochains[1] if dimension >= 1 else None                # redundant. only use for testing. TODO: remove
        self.two_cells = cochains[2] if dimension >= 2 else None            # redundant. only use for testing. TODO: remove

        self.y = y
        
        self._consolidate()
        return
    
    def _consolidate(self):
        for dim in range(self.dimension+1):
            cochain = self.cochains[dim]
            assert cochain.dim == dim
            if dim < self.dimension:
                upper_cochain = self.cochains[dim + 1]
                num_cells_up = upper_cochain.num_cells
                assert num_cells_up is not None
                if 'num_cells_up' in cochain:
                    assert cochain.num_cells_up == num_cells_up
                else:
                    cochain.num_cells_up = num_cells_up
            if dim > 0:
                lower_cochain = self.cochains[dim - 1]
                num_cells_down = lower_cochain.num_cells
                assert num_cells_down is not None
                if 'num_cells_down' in cochain:
                    assert cochain.num_cells_down == num_cells_down
                else:
                    cochain.num_cells_down = num_cells_down
                    
    def to(self, device, **kwargs):
        """Performs tensor dtype and/or device conversion to cochains and label y, if set."""
        # TODO: handle device conversion for specific attributes via `*keys` parameter
        for dim in range(self.dimension + 1):
            self.cochains[dim] = self.cochains[dim].to(device, **kwargs)
        if self.y is not None:
            self.y = self.y.to(device, **kwargs)
        return self

    def get_cochain_params(self,
                           dim : int,
                           max_dim : int=2,
                           include_top_features=True,
                           include_down_features=True,
                           include_boundary_features=True) -> CochainMessagePassingParams:
        """
        Conveniently constructs all necessary input parameters to perform higher-dim
        message passing on the cochain of specified `dim`.

        Args:
            dim: The dimension from which to extract the parameters
            max_dim: The maximum dimension of interest.
                This is only used in conjunction with include_top_features.
            include_top_features: Whether to include the top features from level max_dim+1.
            include_down_features: Include the features for down adjacency
            include_boundary_features: Include the features for the boundary
        Returns:
            An object of type CochainMessagePassingParams
        """
        if dim in self.cochains:
            cells = self.cochains[dim]
            x = cells.x
            # Add up features
            upper_index, upper_features = None, None
            # We also check that dim+1 does exist in the current complex. This cochain might have been
            # extracted from a higher dimensional complex by a batching operation, and dim+1
            # might not exist anymore even though cells.upper_index is present.
            if cells.upper_index is not None and (dim+1) in self.cochains:
                upper_index = cells.upper_index
                if self.cochains[dim + 1].x is not None and (dim < max_dim or include_top_features):
                    upper_features = torch.index_select(self.cochains[dim + 1].x, 0,
                                                        self.cochains[dim].shared_coboundaries)

            # Add down features
            lower_index, lower_features = None, None
            if include_down_features and cells.lower_index is not None:
                lower_index = cells.lower_index
                if dim > 0 and self.cochains[dim - 1].x is not None:
                    lower_features = torch.index_select(self.cochains[dim - 1].x, 0,
                                                        self.cochains[dim].shared_boundaries)
            # Add boundary features
            boundary_index, boundary_features = None, None
            if include_boundary_features and cells.boundary_index is not None:
                boundary_index = cells.boundary_index
                if dim > 0 and self.cochains[dim - 1].x is not None:
                    boundary_features = self.cochains[dim - 1].x

            inputs = CochainMessagePassingParams(x, upper_index, lower_index,
                                               up_attr=upper_features, down_attr=lower_features,
                                               boundary_attr=boundary_features, boundary_index=boundary_index)
        else:
            raise NotImplementedError(
                'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return inputs

    def get_all_cochain_params(self,
                               max_dim:int=2,
                               include_top_features=True,
                               include_down_features=True,
                               include_boundary_features=True) -> List[CochainMessagePassingParams]:
        """Extracts the cochain parameters for message passing on the cochains up to max_dim.

        Args:
            max_dim: The maximum dimension of the complex for which to extract the parameters.
            include_top_features: Whether to include the features from level max_dim+1.
            include_down_features: Include the features for down adjacent cells.
            include_boundary_features: Include the features for the boundary cells.
        Returns:
            A list of elements of type CochainMessagePassingParams.
        """
        all_params = []
        return_dim = min(max_dim, self.dimension)
        for dim in range(return_dim+1):
            all_params.append(self.get_cochain_params(dim, max_dim=max_dim,
                                                    include_top_features=include_top_features,
                                                    include_down_features=include_down_features,
                                                    include_boundary_features=include_boundary_features))
        return all_params

    def get_labels(self, dim=None):
        """Returns target labels.

        If `dim`==k (integer in [0, self.dimension]) then the labels over k-cells are returned.
        In the case `dim` is None the complex-wise label is returned.
        """
        if dim is None:
            y = self.y
        else:
            if dim in self.cochains:
                y = self.cochains[dim].y
            else:
                raise NotImplementedError(
                    'Dim {} is not present in the complex or not yet supported.'.format(dim))
        return y

    def set_xs(self, xs: List[Tensor]):
        """Sets the features of the cochains to the values in the list"""
        assert (self.dimension + 1) >= len(xs)
        for i, x in enumerate(xs):
            self.cochains[i].x = x
            
    @property
    def keys(self):
        """Returns all names of complex attributes."""
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
        return keys
    
    def __getitem__(self, key):
        """Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)
    
    def __contains__(self, key):
        """Returns :obj:`True`, if the attribute :obj:`key` is present in the data."""
        return key in self.keys
    

class ComplexBatch(Complex):
    """Class representing a batch of cochain complexes.

    This is stored as a single cochain complex formed of batched cochains.

    Args:
        cochains: A list of cochain batches that will be put together in a complex batch
        dimension: The dimension of the resulting complex.
        y: A tensor of labels for the complexes in the batch.
        num_complexes: The number of complexes in the batch.
    """
    def __init__(self,
                 *cochains: CochainBatch,
                 dimension: int,
                 y: torch.Tensor = None,
                 num_complexes: int = None):
        super(ComplexBatch, self).__init__(*cochains, y=y)
        self.num_complexes = num_complexes
        self.dimension = dimension

    @classmethod
    def from_complex_list(cls, data_list: List[Complex], follow_batch=[], max_dim: int = 2):
        """Constructs a ComplexBatch from a list of complexes.

        Args:
            data_list: a list of complexes from which the batch is built.
            follow_batch: creates assignment batch vectors for each key in
                :obj:`follow_batch`.
            max_dim: the maximum cochain dimension considered when constructing the batch.
        Returns:
            A ComplexBatch object.
        """

        dimension = max([data.dimension for data in data_list])
        dimension = min(dimension, max_dim)
        cochains = [list() for _ in range(dimension + 1)]
        label_list = list()
        per_complex_labels = True
        for comp in data_list:
            for dim in range(dimension+1):
                if dim not in comp.cochains:
                    # If a dim-cochain is not present for the current complex, we instantiate one.
                    cochains[dim].append(Cochain(dim=dim))
                    if dim-1 in comp.cochains:
                        # If the cochain below exists in the complex, we need to add the number of
                        # boundaries to the newly initialised complex, otherwise batching will not work.
                        cochains[dim][-1].num_cells_down = comp.cochains[dim - 1].num_cells
                else:
                    cochains[dim].append(comp.cochains[dim])
            per_complex_labels &= comp.y is not None
            if per_complex_labels:
                label_list.append(comp.y)

        batched_cochains = [CochainBatch.from_cochain_list(cochain_list, follow_batch=follow_batch)
                          for cochain_list in cochains]
        y = None if not per_complex_labels else torch.cat(label_list, 0)
        batch = cls(*batched_cochains, y=y, num_complexes=len(data_list), dimension=dimension)

        return batch