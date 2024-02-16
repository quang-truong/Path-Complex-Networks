import torch
import numpy as np
from sklearn import metrics as met
from ogb.graphproppred import Evaluator as OGBEvaluator

class Evaluator(object):
    
    def __init__(self, metric, **kwargs):
        if metric == 'isomorphism':
            self.eval_fn = self._isomorphism
            self.eps = kwargs.get('eps', 0.01)
            self.p_norm = kwargs.get('p', 2)
        elif metric == 'accuracy':
            self.eval_fn = self._accuracy
        elif metric == 'mae':
            self.eval_fn = self._mae
        elif metric.startswith('ogbg-mol'):
            self._ogb_evaluator = OGBEvaluator(metric)
            self._key = self._ogb_evaluator.eval_metric
            self.eval_fn = self._ogb
        else:
            raise NotImplementedError('Metric {} is not yet supported.'.format(metric))
    
    def eval(self, input_dict):
        return self.eval_fn(input_dict)
    
    def eval_each_sample(self, input_dict):
        if self.eval_fn == self._mae:
            y_true = input_dict['y_true']
            y_pred = input_dict['y_pred']
            return np.abs(y_true - y_pred)
        else:
            raise NotImplementedError('Evaluation per sample is not yet supported for this metric.')
        
    def _isomorphism(self, input_dict):
        # NB: here we return the failure percentage... the smaller the better!
        preds = input_dict['y_pred']
        assert preds is not None
        assert preds.dtype == np.float64
        preds = torch.tensor(preds, dtype=torch.float64)
        mm = torch.pdist(preds, p=self.p_norm)
        wrong = (mm < self.eps).sum().item()
        metric = wrong / mm.shape[0]
        return metric
    
    def _accuracy(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = np.argmax(input_dict['y_pred'], axis=1)
        assert y_true is not None
        assert y_pred is not None
        metric = met.accuracy_score(y_true, y_pred)
        return metric

    def _mae(self, input_dict, **kwargs):
        y_true = input_dict['y_true']
        y_pred = input_dict['y_pred']
        assert y_true is not None
        assert y_pred is not None
        metric = met.mean_absolute_error(y_true, y_pred)
        return metric
    
    def _ogb(self, input_dict, **kwargs):
        assert 'y_true' in input_dict
        assert input_dict['y_true'] is not None
        assert 'y_pred' in input_dict
        assert input_dict['y_pred'] is not None
        return self._ogb_evaluator.eval(input_dict)[self._key]