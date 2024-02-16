import torch
import os.path as osp

from lib.utils.graph_to_complex import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings, convert_graph_dataset_with_paths
from lib.data.datasets import InMemoryComplexDataset
from ogb.graphproppred import PygGraphPropPredDataset

##################################################
# OGB Complex ####################################
##################################################

class OGBDataset(InMemoryComplexDataset):
    """This is OGB graph-property prediction. This are graph-wise classification tasks."""

    def __init__(self, root, name, use_edge_features=False, transform=None, max_dim = 2,
                 pre_transform=None, pre_filter=None, init_method='sum', complex_type='path',
                 simple=False, include_down_adj=False, n_jobs=2, **kwargs):
        self.name = name
        self._complex_type = complex_type
        self._max_ring_size = kwargs['max_ring_size']
        self._use_edge_features = use_edge_features
        self._simple = simple
        self._n_jobs = n_jobs
        super(OGBDataset, self).__init__(root, transform, pre_transform, pre_filter,
                                         max_dim=max_dim, init_method=init_method, 
                                         include_down_adj=include_down_adj, complex_type=complex_type)
        self.data, self.slices, idx, self.num_tasks = self.load_dataset()
        self.train_ids = idx['train']
        self.val_ids = idx['valid']
        self.test_ids = idx['test']
        
    @property
    def raw_file_names(self):
        name = self.name.replace('-', '_')  # Replacing is to follow OGB folder naming convention
        # The processed graph files are our raw files.
        return [f'{name}/processed/geometric_data_processed.pt']

    @property
    def processed_file_names(self):
        return [f'{self.name}_complex.pt', f'{self.name}_idx.pt', f'{self.name}_tasks.pt']

    def download(self):
        # Instantiating this will download and process the graph dataset.
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)

    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data, slices = torch.load(self.processed_paths[0])
        idx = torch.load(self.processed_paths[1])
        tasks = torch.load(self.processed_paths[2])
        return data, slices, idx, tasks

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
                init_method=self._init_method,
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
        dataset = PygGraphPropPredDataset(self.name, self.raw_dir)
        split_idx = dataset.get_idx_split()
        if self._simple:  # Only retain the top two node/edge features
            print('Using simple features')
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
        
        # NB: the init method would basically have no effect if 
        # we use edge features and do not initialize rings. 
        print(f"Converting the {self.name} dataset to a {self._complex_type} complex...")
        complexes = self.convert_to_complex(dataset)
        
        print(f'Saving processed dataset in {self.processed_paths[0]}...')
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
        
        print(f'Saving idx in {self.processed_paths[1]}...')
        torch.save(split_idx, self.processed_paths[1])
        
        print(f'Saving num_tasks in {self.processed_paths[2]}...')
        torch.save(dataset.num_tasks, self.processed_paths[2])
    
    @property
    def processed_dir(self):
        """Overwrite to change name based on edge and simple feats"""
        directory = super(OGBDataset, self).processed_dir
        suffix1 = f"-max_ring_{self._max_ring_size}" if self._complex_type == 'cell' else ""
        suffix2 = "-use_edge_features" if self._use_edge_features else ""
        suffix3 = f"-down_adj" if self.include_down_adj else ""
        suffix4 = "-simple" if self._simple else ""
        return directory + suffix1 + suffix2 + suffix3 + suffix4

##################################################
# OGB Graph ######################################
##################################################


def load_ogb_graph_dataset(root, name):
    raw_dir = osp.join(root, 'raw')
    dataset = PygGraphPropPredDataset(name, raw_dir)
    idx = dataset.get_idx_split()
    num_classes = dataset.num_tasks
    num_features = dataset[0].x.shape[1]

    return dataset, idx['train'], idx['valid'], idx['test'], num_classes, num_features
