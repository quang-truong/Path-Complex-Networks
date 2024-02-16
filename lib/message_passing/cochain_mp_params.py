"""
Based on https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/nn/conv/message_passing.py

MIT License

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

from torch import Tensor
from torch_geometric.typing import Adj

class CochainMessagePassingParams:
    """A helper class storing the parameters to be supplied to the propagate function.

    This object stores the equivalent of the `x` and `edge_index` objects from PyTorch Geometric.
    TODO: The boundary_index and boundary_attr as well as other essential parameters are
          currently passed as keyword arguments. Special parameters should be created.
    Args:
        x: The features of the cochain where message passing will be performed.
        up_index: The index for the upper adjacencies of the cochain.
        down_index: The index for the lower adjacencies of the cochain.
    """
    def __init__(self, x: Tensor, up_index: Adj = None, down_index: Adj = None, **kwargs):
        self.x = x
        self.up_index = up_index
        self.down_index = down_index
        self.kwargs = kwargs
        if 'boundary_index' in self.kwargs:
            self.boundary_index = self.kwargs['boundary_index']
        else:
            self.boundary_index = None
        if 'boundary_attr' in self.kwargs:
            self.boundary_attr = self.kwargs['boundary_attr']
        else:
            self.boundary_attr = None