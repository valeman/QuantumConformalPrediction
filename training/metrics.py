import torch

import torch.nn as nn
class NegativeLogSumCriterion(nn.Module):
    def __init__(self):
        super(NegativeLogSumCriterion, self).__init__()

    def forward(self, input_tensor):
        clamped_tensor = torch.clamp(input_tensor, min=0.01)
        log_tensor = torch.log(clamped_tensor)
        sum_log = torch.sum(log_tensor)
        return -sum_log