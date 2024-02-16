import torch

from typing import Callable, Optional
from lib.message_passing.cochain_mp_params import CochainMessagePassingParams
from lib.layers.reduce_conv import InitReduceConv
from torch_geometric.nn.inits import reset
from abc import ABC, abstractmethod
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class AbstractEmbedVEWithReduce(torch.nn.Module, ABC):              # TODO: Need to extend to k-dim
    
    def __init__(self,
                 v_embed_layer: Callable,
                 e_embed_layer: Optional[Callable],
                 init_reduce: InitReduceConv):
        """

        Args:
            v_embed_layer: Layer to embed the integer features of the vertices
            e_embed_layer: Layer (potentially None) to embed the integer features of the edges.
            init_reduce: Layer to initialise the 2D cell features and potentially the edge features.
        """
        super(AbstractEmbedVEWithReduce, self).__init__()
        self.v_embed_layer = v_embed_layer
        self.e_embed_layer = e_embed_layer
        self.init_reduce = init_reduce
    
    @abstractmethod
    def _prepare_v_inputs(self, v_params):
        pass
    
    @abstractmethod
    def _prepare_e_inputs(self, e_params):
        pass
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        v_params = cochain_params[0]
        e_params = cochain_params[1] if len(cochain_params) >= 2 else None
        remain_params = [cochain_params[i] for i in range(2, len(cochain_params))] if len(cochain_params) >= 3 else None

        vx = self.v_embed_layer(self._prepare_v_inputs(v_params))
        out = [vx]

        if e_params is None:
           assert remain_params is None
           return out

        # edge feature is aggregated from node feature 
        reduced_ex = self.init_reduce(vx, e_params.boundary_index)

        if e_params.x is not None and self.e_embed_layer is not None:           # TODO: edge feature always exists, but may not be encoded by edge_attr if using gudhi
            ex = self.e_embed_layer(self._prepare_e_inputs(e_params))           # consider remove e_params.x from if condition since it always exists.
            # The output of this should be the same size as the vertex features.
            assert ex.size(1) == vx.size(1)
        else:
            ex = reduced_ex
        out.append(ex)

        if remain_params is not None:
            # We divide by two in case this was obtained from node aggregation.
            # The division should not do any harm if this is an aggregation of learned embeddings.
            for i in range(2, len(cochain_params)):
                out.append(self.init_reduce(reduced_ex, remain_params[i-2].boundary_index))
                reduced_ex = out[-1]

        return out
    
    def reset_parameters(self):
        reset(self.v_embed_layer)
        reset(self.e_embed_layer)

    
class EmbedVEWithReduce(AbstractEmbedVEWithReduce):

    def __init__(self,
                 v_embed_layer: torch.nn.Embedding,
                 e_embed_layer: Optional[torch.nn.Embedding],
                 init_reduce: InitReduceConv):
        super(EmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)
        
    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        assert v_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return v_params.x.squeeze(1).to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert e_params.x.dim() == 2
        assert e_params.x.size(1) == 1
        # The embedding layer expects integers so we convert the tensor to int.
        return e_params.x.squeeze(1).to(dtype=torch.long)
    

class OGBEmbedVEWithReduce(AbstractEmbedVEWithReduce):
    
    def __init__(self,
                 v_embed_layer: AtomEncoder,
                 e_embed_layer: Optional[BondEncoder],
                 init_reduce: InitReduceConv):
        super(OGBEmbedVEWithReduce, self).__init__(v_embed_layer, e_embed_layer, init_reduce)

    def _prepare_v_inputs(self, v_params):
        assert v_params.x is not None
        assert v_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return v_params.x.to(dtype=torch.long)
    
    def _prepare_e_inputs(self, e_params):
        assert self.e_embed_layer is not None
        assert e_params.x.dim() == 2
        # Inputs in ogbg-mol* datasets are already long.
        # This is to test the layer with other datasets.
        return e_params.x.to(dtype=torch.long)