import torch
import os.path as osp

from lib.utils.graph_to_complex import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings, convert_graph_dataset_with_paths
from lib.data.datasets import InMemoryComplexDataset
from torch_geometric.datasets import ZINC

##################################################
# ZINC Complex ###################################
##################################################

class ZincDataset(InMemoryComplexDataset):
    """This is ZINC from the Benchmarking GNNs paper. This is a graph regression task."""

    def __init__(self, root, use_edge_features=False, init_method = 'sum', complex_type = 'path', 
                 max_dim = 2, transform=None, pre_transform=None, pre_filter=None, subset=True, 
                 include_down_adj=False, n_jobs=2, **kwargs):
        self.name = 'ZINC'
        self._complex_type = complex_type
        self._max_ring_size = kwargs['max_ring_size']
        self._use_edge_features = use_edge_features
        self._subset = subset
        self._n_jobs = n_jobs
        super(ZincDataset, self).__init__(root, transform, pre_transform, pre_filter, max_dim=max_dim, include_down_adj=include_down_adj,
                                          init_method = init_method, complex_type=complex_type, num_classes=1)

        self.data, self.slices, idx = self.load_dataset()
        self.train_ids = idx[0]
        self.val_ids = idx[1]
        self.test_ids = idx[2]

        self.num_node_type = 28
        self.num_edge_type = 4

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        name = self.name
        return [f'{name}_complex.pt', f'{name}_idx.pt']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        ZINC(self.raw_dir, subset=self._subset)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        return data, slices, idx
    
    def convert_to_complex(self, data):
        if self._complex_type == 'simplicial':
            complexes, _, _, = convert_graph_dataset_with_gudhi(        # this function doesn't encode edge_attr, so must set use_edge_features False
                data,
                expansion_dim = self._max_dim,
                include_down_adj = self.include_down_adj,
                init_method = self._init_method
            )
        elif self._complex_type == 'cell':
            complexes, _, _ = convert_graph_dataset_with_rings(
                data,
                max_ring_size=self._max_ring_size,
                include_down_adj=self.include_down_adj,
                init_edges=self._use_edge_features,
                init_rings=False,
                n_jobs=self._n_jobs)
        elif self._complex_type == 'path':
            complexes, _, _ = convert_graph_dataset_with_paths(
                data,
                max_k = self._max_dim,
                include_down_adj=self.include_down_adj,
                init_edges=self._use_edge_features,
                init_high_order_paths=False,
                init_method = self._init_method,
                n_jobs=self._n_jobs)
        else:
            raise ValueError("Complex type not supported for this dataset")
        return complexes

    def process(self):
        # At this stage, the graph dataset is already downloaded and processed
        print(f"Processing {self._complex_type} complex dataset for {self.name}")
        train_data = ZINC(self.raw_dir, subset=self._subset, split='train')
        val_data = ZINC(self.raw_dir, subset=self._subset, split='val')
        test_data = ZINC(self.raw_dir, subset=self._subset, split='test')

        data_list = []
        idx = []
        start = 0
        print(f"Converting the train dataset to a {self._complex_type} complex...")
        train_complexes = self.convert_to_complex(train_data)
        data_list += train_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print(f"Converting the validation dataset to a {self._complex_type} complex...")
        val_complexes = self.convert_to_complex(val_data)
        data_list += val_complexes
        idx.append(list(range(start, len(data_list))))
        start = len(data_list)
        print(f"Converting the test dataset to a {self._complex_type} complex...")
        test_complexes = self.convert_to_complex(test_data)
        data_list += test_complexes
        idx.append(list(range(start, len(data_list))))

        path = self.processed_paths[0]
        print(f'Saving processed dataset in {path}....')
        torch.save(self.collate(data_list, self.max_dim), path)
        
        path = self.processed_paths[1]
        print(f'Saving idx in {path}....')
        torch.save(idx, path)

    @property
    def processed_dir(self):
        """Overwrite to change name based on edges"""

        directory = super(ZincDataset, self).processed_dir
        suffix0 = "-full" if self._subset is False else ""
        suffix1 = f"-max_ring_{self._max_ring_size}" if self._complex_type == 'cell' else ""
        suffix2 = "-use_edge_features" if self._use_edge_features else ""
        suffix3 = f"-down_adj" if self.include_down_adj else ""
        return directory + suffix0 + suffix1 + suffix2 + suffix3



##################################################
# ZINC Graph #####################################
##################################################


def load_zinc_graph_dataset(root, subset=True):
    raw_dir = osp.join(root, 'ZINC', 'raw')

    train_data = ZINC(raw_dir, subset=subset, split='train')
    val_data = ZINC(raw_dir, subset=subset, split='val')
    test_data = ZINC(raw_dir, subset=subset, split='test')
    data = train_data + val_data + test_data

    if subset:
        assert len(train_data) == 10000
        assert len(val_data) == 1000
        assert len(test_data) == 1000
    else:
        assert len(train_data) == 220011
        assert len(val_data) == 24445
        assert len(test_data) == 5000

    idx = []
    start = 0
    idx.append(list(range(start, len(train_data))))
    start = len(train_data)
    idx.append(list(range(start, start + len(val_data))))
    start = len(train_data) + len(val_data)
    idx.append(list(range(start, start + len(test_data))))
    num_classes = 1
    num_features = train_data[0].x.shape[1]
    return data, idx[0], idx[1], idx[2], num_classes, num_features


