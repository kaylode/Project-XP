import torch
import torch.nn as nn
import torch.utils.data as data

from metrics.classification import AccuracyMetric



class BaseModel(nn.Module):
    def __init__(self,
                metrics = AccuracyMetric(),
                device = None,
                freeze = False):

        super(BaseModel, self).__init__()
        
        self.device = device
        self.freeze = freeze
        self.metrics = metrics
        if not isinstance(metrics, list):
            self.metrics = [metrics,]


