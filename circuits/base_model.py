import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError("Each model must implement the forward method.")