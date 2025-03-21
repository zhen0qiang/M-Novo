import torch.nn.functional as F


class Loss:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, *args, **kwargs):
        if self.mode == 'NLLLoss':
            return self.nll_loss(*args, **kwargs)
        elif self.mode == 'CrossEntropyLoss':
            assert len(args) == 2, "CrossEntropyLoss takes two arguments: input and target"
            return self.cross_entropy_loss(*args, **kwargs)
        else:
            raise ValueError("Invalid loss mode")
    
    def cross_entropy_loss(self, output, target):
        assert len(output.shape) == 2 or len(output.shape) == 3, "CrossEntropyLoss input should be 2D or 3D"
        assert len(target.shape) == 1 or len(target.shape) == 2, "CrossEntropyLoss target should be 1D or 2D"
        
        if len(output.shape) == 3:
            output = output.view(-1, output.size(2))
        if len(target.shape) == 2:
            target = target.view(-1)
            
        return F.cross_entropy(output, target)