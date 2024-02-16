import torch

cls_criterion = torch.nn.CrossEntropyLoss()
bicls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.L1Loss()
msereg_criterion = torch.nn.MSELoss()

def get_criterion(task_type: str):
    if task_type == 'classification':
        loss_fn = cls_criterion
    elif task_type == 'bin_classification':
        loss_fn = bicls_criterion
    elif task_type == 'regression':
        loss_fn = reg_criterion
    elif task_type == 'mse_regression':
        loss_fn = msereg_criterion
    elif task_type == 'isomorphism':
        loss_fn = None
    else:
        raise NotImplementedError('Training on task type {} not yet supported.'.format(task_type))
    return loss_fn