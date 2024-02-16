import torch
from torch_scatter import scatter

class InitReduceConv(torch.nn.Module):

    def __init__(self, reduce='add'):
        """

        Args:
            reduce (str): Way to aggregate boundaries. Can be "sum, add, mean, min, max"
        """
        super(InitReduceConv, self).__init__()
        self.reduce = reduce

    def forward(self, boundary_x, boundary_index, out_size = None):
        features = boundary_x.index_select(0, boundary_index[0])
        if out_size is None:
            out_size = boundary_index[1, :].max() + 1
        return scatter(features, boundary_index[1], dim=0, dim_size=out_size, reduce=self.reduce)