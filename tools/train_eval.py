import os
import sys
import subprocess
import copy
import torch
import torch.optim as optim
import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from lib.helpers.data_helpers import get_dataloaders
from lib.helpers.lr_scheduler import get_lr_scheduler
from lib.utils.random_seed import set_random_seed
from lib.helpers.trainer import Trainer
from lib.helpers.eval_helpers import extract_results_molecular_datasets, extract_results_tu_datasets, extract_results_sr_datasets
from tools.parser import get_parser, validate_args
from lib.helpers.model_helpers import get_complex_model, get_graph_model
from lib.utils.sr_utils import sr_families

import wandb


def main(args):
    """The common training and evaluation script used by all the experiments."""
    
    # Set random seed (default 43)
    set_random_seed(args.seed)

    # get timestamp to name experiment
    time_stamp = datetime.datetime.now().isoformat(timespec='seconds')

    # set device
    device = torch.device("cuda:" + str(args.device)) if (torch.cuda.is_available() and args.device != -1) else torch.device("cpu")

    # set result folder to store sub-folder with different seeds
    if not args.sweep:
        result_folder = os.path.join(args.result_folder, args.dataset, 'runs', f'{time_stamp}-{args.exp_name}')
    else:
        result_folder = os.path.join(args.result_folder, args.dataset, 'sweeps', f'{time_stamp}-{args.exp_name}')

    # Set double precision for SR experiments
    if args.task_type == 'isomorphism':
        assert args.dataset == "SR-GRAPHS"                                      # we will set each individual dataset later
        torch.set_default_dtype(torch.float64)

    if not args.debug:
        # Init MLOps
        wandb.init(
            project="PCN",
            config = vars(args)
        )

        wandb.run.name = f'{time_stamp}-{args.exp_name}'

    if args.task_type == 'isomorphism':         # SR-GRAPHS
        results = train_with_different_sr_families(args, result_folder, device)
        args.dataset = "SR-GRAPHS"
    else:
        if args.folds is None:  # ZINC and OGBG datasets
            results = train_with_different_seeds(args, result_folder, device)
        else:                   # when fold is present (TU dataset and CSL)
            results = train_with_different_folds(args, result_folder, device)

    # Get final results across seeds
    if (args.dataset in ['ZINC', 'ZINC-FULL', 'MOLHIV', 'MOLPCBA', 'MOLTOX21', 'MOLTOXCAST', 'MOLMUV',
                'MOLBACE', 'MOLBBBP', 'MOLCLINTOX', 'MOLSIDER', 'MOLESOL',
                'MOLFREESOLV', 'MOLLIPO']):
        extract_results_molecular_datasets(args, results, result_folder)
    elif (args.dataset in ['IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K', 'PROTEINS', 
                            'NCI1', 'NCI109', 'PTC', 'MUTAG']):
        extract_results_tu_datasets(args, results, result_folder)
    elif (args.dataset == 'SR-GRAPHS'):
        extract_results_sr_datasets(args, results, result_folder)
    
    if not args.debug:
        # End Wandb
        wandb.finish()

    return

def train_with_different_seeds(args, result_folder, device):
    results = []            # store statistics across different seeds
    # iterate through seeds
    for seed in range(args.start_seed, args.stop_seed + 1):
        # Data loading and Model initialization
        if args.model.startswith('gin'):
            train_loader, valid_loader, test_loader, num_classes, num_features = get_dataloaders(args, fold = None)
            model = get_graph_model(args, num_features, num_classes, device)
        else:
            train_loader, valid_loader, test_loader, dataset = get_dataloaders(args, fold = None)
            model = get_complex_model(args, dataset, device)

        # instantiate optimiser
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # instantiate learning rate decay
        scheduler = get_lr_scheduler(args, optimizer)
        trainer = Trainer(model, args, train_loader, valid_loader, test_loader, optimizer, scheduler, result_folder, seed, None, device)

        # (!) start training/evaluation
        curves = trainer.train()
        results.append(curves)
    return results

def train_with_different_folds(args, result_folder, device):
    results = []            # store statistics across different seeds
    for fold in range(args.folds):
        # Data loading and Model initialization
        if args.model.startswith('gin'):
            train_loader, valid_loader, test_loader, num_classes, num_features = get_dataloaders(args, fold = fold)
            model = get_graph_model(args, num_features, num_classes, device)
        else:
            train_loader, valid_loader, test_loader, dataset = get_dataloaders(args, fold = fold)
            model = get_complex_model(args, dataset, device)
        
        # instantiate optimiser
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

        # instantiate learning rate decay
        scheduler = get_lr_scheduler(args, optimizer)
        trainer = Trainer(model, args, train_loader, valid_loader, test_loader, optimizer, scheduler, result_folder, None, fold, device)

        # (!) start training/evaluation
        curves = trainer.train()
        results.append(curves)
    return results

def train_with_different_sr_families(args, result_folder, device):
    all_results = []
    families = sr_families()
    for f in families:
        results = []
        print("==========================================================")
        print(f"Family {f}")
        args.dataset = f
        results = train_with_different_seeds(args, os.path.join(result_folder, 'SR-GRAPHS', args.dataset), device)
        all_results.append(results)
    return all_results
        

if __name__ == "__main__":
    passed_args = sys.argv[1:]
    # Extract the commit sha so we can check the code that was used for each experiment
    sha = subprocess.check_output(["git", "describe", "--always"]).strip().decode()
    passed_args = copy.copy(passed_args) + ['--sha', sha]
    parser = get_parser()
    args = parser.parse_args(copy.copy(passed_args))
    assert args.stop_seed >= args.start_seed
    validate_args(args)
    main(args)
