import torch
import numpy as np

from lib.helpers.criterion import get_criterion
from lib.helpers.evaluator import Evaluator
from lib.helpers.save_helpers import get_checkpoint_state, save_checkpoint, load_checkpoint
from lib.utils.log_utils import args_to_string
from lib.helpers.model_helpers import compute_params

from tqdm import tqdm
import logging
import os
import wandb

from lib.data.complex import ComplexBatch

class Trainer(object):
    def __init__(self, model, args, train_loader, valid_loader, test_loader, optimizer, scheduler, result_folder, train_seed, fold, device):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_seed = train_seed if train_seed is not None else 0
        self.fold = fold
        self.task_type = args.task_type
        self.device = device
        self.loss_fn = get_criterion(self.task_type)

        # set up evaluator
        self.evaluator = Evaluator(self.args.eval_metric, eps=self.args.iso_eps)
        if (not args.eval_only):
            # keep track statistics
            self.best_val_epoch = 0
            self.valid_curve = []
            self.test_curve = []
            self.train_curve = []
            self.train_loss_curve = []
            self.params = []

            # Create results folder for this seed only
            if (not self.args.debug):
                new_folder = os.path.join(result_folder, f'seed_{self.train_seed}')
                self.filename = os.path.join(new_folder, f'results-seed_{self.train_seed}.txt')
                if fold is not None:
                    new_folder = os.path.join(new_folder, f'fold_{fold}')
                    self.filename = os.path.join(new_folder, f'results-seed_{self.train_seed}-fold_{fold}.txt')
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                self.best_epoch_pt = os.path.join(new_folder, 'best_weights.pt')

            # Print Run's Header Information
            print("==========================================================")
            print("Using device", str(device))
            print(f"Fold: {fold}")
            print(f"Random Seed: {args.seed}")
            print(f"Train Seed: {self.train_seed}")
            print("======================== Args ===========================")
            print(args)
            print("=========================================================")

            compute_params(model)

    def load_saved_model(self, weights_file):
        load_checkpoint(self.model, self.optimizer, weights_file)
    
    def log_curves(self, log_dict, name, curves, suffix, is_best = False):
        prefix = 'Best ' if is_best else ''
        log_dict[f'{prefix}Train {name} {suffix}'] = curves[0]
        log_dict[f'{prefix}Val {name} {suffix}'] = curves[1]
        if len(curves) == 3:
            log_dict[f'{prefix}Test {name} {suffix}'] = curves[2]
        return log_dict

    def train_one_epoch(self):
        curve = list()
        self.model.train()
        num_skips = 0                   # Keep track number of skipped graphs due to drop_last
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training iteration")):
            batch = batch.to(self.device)
            if isinstance(batch, ComplexBatch):
                num_samples = batch.cochains[0].x.size(0)
                for dim in range(1, batch.dimension+1):
                    num_samples = min(num_samples, batch.cochains[dim].num_cells)
            else:
                # This is graph.
                num_samples = batch.x.size(0)

            if num_samples <= 1:
                # Skip batch if it only comprises one sample (could cause problems with BN)
                num_skips += 1
                if float(num_skips) / len(self.train_loader) >= 0.25:
                    logging.warning("Warning! 25% of the batches were skipped this epoch")
                continue
            
            # (DEBUG)
            if num_samples < 10:
                logging.warning("Warning! BatchNorm applied on a batch "
                                "with only {} samples".format(num_samples))

            self.optimizer.zero_grad()
            pred = self.model(batch)
            if isinstance(self.loss_fn, torch.nn.CrossEntropyLoss):             # Classification problem
                targets = batch.y.view(-1,)
            else:                                                               # Regression or Isomorphism problem
                targets = batch.y.to(torch.float32).view(pred.shape)

            # In some ogbg-mol* datasets we may have null targets.
            # When the cross entropy loss is used and targets are of shape (N,)
            # the maks is broadcasted automatically to the shape of the predictions.
            mask = ~torch.isnan(targets)
            loss = self.loss_fn(pred[mask], targets[mask])

            loss.backward()
            self.optimizer.step()
            curve.append(loss.detach().cpu().item())
        epoch_train_loss = float(np.mean(curve))
        return epoch_train_loss
    
    def train(self):
        best_val_epoch = 0
        valid_curve = []
        test_curve = []
        train_curve = []

        suffix = f'(Seed {self.train_seed} - Fold {self.fold})'

        if not self.args.untrained:
            for epoch in range(1, self.args.epochs + 1):
                log_dict = {}
                # perform one epoch
                print("=====Epoch {}".format(epoch))

                # reset numpy seed.
                # ref: https://github.com/pytorch/pytorch/issues/5059
                np.random.seed(np.random.get_state()[1][0] + epoch)

                print('Training...')
                epoch_train_loss = self.train_one_epoch()

                # evaluate model on train_loader, valid_loader, and test_loader
                print('Evaluating...')
                if epoch == 1 or epoch % self.args.train_eval_period == 0:
                    train_perf, _ = self.eval(self.train_loader)
                
                valid_perf, epoch_val_loss = self.eval(self.valid_loader)
                

                if self.test_loader is not None:
                    test_perf, epoch_test_loss = self.eval(self.test_loader)
                else:
                    test_perf = np.nan
                    epoch_test_loss = np.nan

                train_curve.append(train_perf)
                valid_curve.append(valid_perf)
                test_curve.append(test_perf)

                print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
                    f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}'
                    f' | Test Loss {epoch_test_loss:.3f}')

                # decay learning rate
                if self.scheduler is not None:
                    if self.args.lr_scheduler == 'ReduceLROnPlateau':
                        self.scheduler.step(valid_perf)
                        # We use a strict inequality here like in the benchmarking GNNs paper code
                        # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py#L217
                        if self.args.early_stop and self.optimizer.param_groups[0]['lr'] < self.args.lr_scheduler_min:
                            print("\n!! The minimum learning rate has been reached.")
                            break
                    else:
                        self.scheduler.step()
                perf_curves = (train_perf, valid_perf, test_perf) if not np.isnan(test_perf) else (train_perf, valid_perf)
                loss_curves = (epoch_train_loss, epoch_val_loss, epoch_test_loss) if not np.isnan(epoch_test_loss) else (epoch_train_loss, epoch_val_loss)
                log_dict = self.log_curves(log_dict, 'Performance', perf_curves, suffix, is_best = False)
                log_dict = self.log_curves(log_dict, 'Loss', loss_curves, suffix, is_best = False)
                log_dict[f'Epoch {suffix}'] = epoch

                # update best_val_epoch and save weights
                if ((valid_perf <= valid_curve[best_val_epoch] and self.args.minimize) or (valid_perf >= valid_curve[best_val_epoch] and not self.args.minimize)):
                    best_val_epoch = epoch - 1              # since epoch start at 1
                    log_dict = self.log_curves(log_dict, 'Performance', perf_curves, suffix, is_best = True)
                    if (not self.args.debug):
                        save_checkpoint(get_checkpoint_state(self.model, self.optimizer, epoch), self.best_epoch_pt)
                if (not self.args.debug):
                    wandb.log(log_dict)
        else:
            train_curve.append(np.nan)
            valid_curve.append(np.nan)
            test_curve.append(np.nan)

        print('Final Evaluation...')
        final_train_perf = np.nan
        final_val_perf = np.nan
        final_test_perf = np.nan
        if not self.args.dataset.startswith('sr'):
            final_train_perf, _ = self.eval(self.train_loader)
            final_val_perf, _ = self.eval( self.valid_loader)
        if self.test_loader is not None:
            final_test_perf, _ = self.eval(self.test_loader)

        msg = (
        f'========== Result ============\n'
        f'Dataset:        {self.args.dataset}\n'
        f'------------ Best epoch -----------\n'
        f'Train:          {train_curve[best_val_epoch]}\n'
        f'Validation:     {valid_curve[best_val_epoch]}\n'
        f'Test:           {test_curve[best_val_epoch]}\n'
        f'Best epoch:     {best_val_epoch}\n'
        '------------ Last epoch -----------\n'
        f'Train:          {final_train_perf}\n'
        f'Validation:     {final_val_perf}\n'
        f'Test:           {final_test_perf}\n'
        '-------------------------------\n\n')
        print(msg)

        msg += args_to_string(self.args)
        

        # save results
        curves = {
            'train': train_curve,
            'val': valid_curve,
            'test': test_curve,
            'last_val': final_val_perf,
            'last_test': final_test_perf,
            'last_train': final_train_perf,
            'best': best_val_epoch
        }
        
        # dump results.txt
        if (not self.args.debug):
            with open(self.filename, 'w') as handle:
                handle.write(msg)
            wandb.save(self.filename)
        return curves


    def eval(self, loader, return_perf_each_sample = False):
        """
            Evaluates a model over all the batches of a data loader.
        """
        loss_fn = get_criterion(self.task_type)
        self.model.eval()
        y_true = []
        y_pred = []
        losses = []
        for step, batch in enumerate(tqdm(loader, desc="Eval iteration")):
            
            # Cast features to double precision if that is used
            if torch.get_default_dtype() == torch.float64:
                for dim in range(batch.dimension + 1):
                    batch.cochains[dim].x = batch.cochains[dim].x.double()
                    assert batch.cochains[dim].x.dtype == torch.float64, batch.cochains[dim].x.dtype

            batch = batch.to(self.device)
            with torch.no_grad():
                pred = self.model(batch)
                
                if self.task_type != 'isomorphism':
                    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                        targets = batch.y.view(-1,)
                        y_true.append(batch.y.detach().cpu())
                    else:
                        targets = batch.y.to(torch.float32).view(pred.shape)
                        y_true.append(batch.y.view(pred.shape).detach().cpu())
                    mask = ~torch.isnan(targets)  # In some ogbg-mol* datasets we may have null targets.
                    loss = loss_fn(pred[mask], targets[mask])
                    losses.append(loss.detach().cpu().item())
                else:
                    assert self.loss_fn is None
                
            y_pred.append(pred.detach().cpu())

        y_true = torch.cat(y_true, dim=0).numpy()  if len(y_true) > 0 else None
        y_pred = torch.cat(y_pred, dim=0).numpy()

        input_dict = {'y_pred': y_pred, 'y_true': y_true}
        mean_loss = float(np.mean(losses)) if len(losses) > 0 else np.nan

        if return_perf_each_sample:
            return self.evaluator.eval_each_sample(input_dict), self.evaluator.eval(input_dict), mean_loss
        else:
            return self.evaluator.eval(input_dict), mean_loss