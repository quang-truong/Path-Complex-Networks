import os
import torch
import pickle
import numpy as np
from definitions import ROOT_DIR

from lib.utils.tu_utils import load_data, S2V_to_PyG, get_fold_indices
from lib.utils.graph_to_complex import convert_graph_dataset_with_gudhi, convert_graph_dataset_with_rings, convert_graph_dataset_with_paths
from lib.data.datasets import InMemoryComplexDataset

class TUDataset(InMemoryComplexDataset):
    """A dataset of complexes obtained by lifting graphs from TUDatasets."""

    def __init__(self, root, name, max_dim=2, max_ring_size=None,num_classes=2, num_node_labels = 3,
                 degree_as_tag=False, disable_one_hot=False, fold=0, init_method='sum', complex_type='path', 
                 seed=0, include_down_adj=False, n_jobs=2, **kwargs):
        self.name = name
        self.degree_as_tag = degree_as_tag
        self.num_node_labels = num_node_labels
        self.disable_one_hot = disable_one_hot
        self._n_jobs = n_jobs
        self._max_ring_size = max_ring_size
        assert self._max_ring_size is None or self._max_ring_size > 3

        if complex_type == 'cell':
            assert max_dim == 2

        super(TUDataset, self).__init__(root, max_dim=max_dim, num_classes=num_classes,
            init_method=init_method, include_down_adj=include_down_adj, complex_type=complex_type)

        self.data, self.slices = torch.load(self.processed_paths[0])
            
        self.fold = fold
        self.seed = seed
        train_filename = os.path.join(self.raw_dir, '10fold_idx', 'train_idx-{}.txt'.format(fold + 1))  
        test_filename = os.path.join(self.raw_dir, '10fold_idx', 'test_idx-{}.txt'.format(fold + 1))
        if os.path.isfile(train_filename) and os.path.isfile(test_filename):
            # NB: we consider the loaded test indices as val_ids ones and set test_ids to None
            #     to make it more convenient to work with the training pipeline
            self.train_ids = np.loadtxt(train_filename, dtype=int).tolist()
            self.val_ids = np.loadtxt(test_filename, dtype=int).tolist()
        else:
            train_ids, val_ids = get_fold_indices(self, self.seed, self.fold)
            self.train_ids = train_ids
            self.val_ids = val_ids
        self.test_ids = None


    @property
    def raw_file_names(self):
        # The processed graph files are our raw files.
        # They are obtained when running the initial data conversion S2V_to_PyG. 
        return ['{}_graph_list_degree_as_tag_{}_disable_one_hot_{}.pkl'.format(self.name, self.degree_as_tag, self.disable_one_hot)]

    @property
    def processed_file_names(self):
        return ['{}_complex_list.pt'.format(self.name)]
    
    def download(self):
        # This will process the raw data into a list of PyG Data objs.
        data, num_classes, num_node_labels = load_data(self.raw_dir, self.name, self.degree_as_tag, self.disable_one_hot)
        self._num_classes = num_classes
        self.num_node_labels = num_node_labels
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(self.raw_paths[0], 'wb') as handle:
            pickle.dump(graph_list, handle)

    def convert_to_complex(self, data):
        init_edges = False if self.disable_one_hot else True
        init_rings = False if self.disable_one_hot else True
        init_high_order_paths = False if self.disable_one_hot else True
        if self._complex_type == 'simplicial':
            complexes, _, _, = convert_graph_dataset_with_gudhi(        
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
                init_method = self._init_method,
                init_edges=init_edges,
                init_rings=init_rings,
                n_jobs=self._n_jobs)
        elif self._complex_type == 'path':
            complexes, _, _ = convert_graph_dataset_with_paths(
                data,
                max_k = self._max_dim,
                include_down_adj=self.include_down_adj,
                init_edges=init_edges,
                init_high_order_paths=init_high_order_paths,
                init_method = self._init_method,
                n_jobs=self._n_jobs)
        else:
            raise ValueError("Complex type not supported for this dataset")
        return complexes

    def process(self):
        with open(self.raw_paths[0], 'rb') as handle:
            graph_list = pickle.load(handle)
                
        print(f"Converting the dataset to a {self._complex_type} complex...")
        complexes = self.convert_to_complex(graph_list)

        print(f'Saving processed dataset in {self.processed_paths[0]}....')
        torch.save(self.collate(complexes, self.max_dim), self.processed_paths[0])
    
    @property
    def processed_dir(self):
        """This is overwritten, so the cellular complex data is placed in another folder"""
        directory = super(TUDataset, self).processed_dir
        suffix = f"-max_ring_{self._max_ring_size}" if self._complex_type == 'cell' else ""
        suffix += f"-down_adj" if self.include_down_adj else ""
        suffix += f"-disable_one_hot" if self.disable_one_hot else ""
        return directory + suffix

##################################################
# TU Graph #######################################
##################################################

def load_tu_graph_dataset(name, root=os.path.join(ROOT_DIR, 'datasets'), degree_as_tag=False, disable_one_hot=False, fold=0, seed=0):
    raw_dir = os.path.join(root, name, 'raw')
    load_from = os.path.join(raw_dir, '{}_graph_list_degree_as_tag_{}_disable_one_hot_{}.pkl'.format(name, degree_as_tag, disable_one_hot))
    if os.path.isfile(load_from):
        with open(load_from, 'rb') as handle:
            graph_list = pickle.load(handle)
    else:
        data, num_classes, num_node_labels = load_data(raw_dir, name, degree_as_tag, disable_one_hot)
        print('Converting graph data into PyG format...')
        graph_list = [S2V_to_PyG(datum) for datum in data]
        with open(load_from, 'wb') as handle:
            pickle.dump(graph_list, handle)
    train_filename = os.path.join(raw_dir, '10fold_idx', 'train_idx-{}.txt'.format(fold + 1))  
    test_filename = os.path.join(raw_dir, '10fold_idx', 'test_idx-{}.txt'.format(fold + 1))
    if os.path.isfile(train_filename) and os.path.isfile(test_filename):
        # NB: we consider the loaded test indices as val_ids ones and set test_ids to None
        #     to make it more convenient to work with the training pipeline
        train_ids = np.loadtxt(train_filename, dtype=int).tolist()
        val_ids = np.loadtxt(test_filename, dtype=int).tolist()
    else:
        train_ids, val_ids = get_fold_indices(graph_list, seed, fold)
    test_ids = None

    if (name == 'IMDBBINARY'):
        num_classes = 2
        num_node_labels = 65
    elif (name == 'IMDBMULTI'):
        num_classes = 3
        num_node_labels = 59
    elif (name == 'REDDITBINARY'):
        num_classes = 2
        num_node_labels = 1
    elif (name == 'REDDITMULTI5K'):
        num_classes = 5
        num_node_labels = 1
    elif (name == 'PROTEINS'):
        num_classes = 2
        num_node_labels = 3
    elif (name == 'NCI1'):
        num_classes = 2
        num_node_labels = 37
    elif (name == 'NCI109'):
        num_classes = 2
        num_node_labels = 38
    elif (name == 'PTC'):
        num_classes = 2
        num_node_labels = 19
    elif (name == 'MUTAG'):
        num_classes = 2
        num_node_labels = 7
    if disable_one_hot:
        num_features = 1
    else:
        num_features = graph_list[0].x.shape[1]
    return graph_list, train_ids, val_ids, test_ids, num_classes, num_features, num_node_labels