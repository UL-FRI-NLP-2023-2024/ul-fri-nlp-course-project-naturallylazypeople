from trainers.trainer_base import TrainerBase
from copy import deepcopy

class BitFitTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        model = deepcopy(kwargs.pop("model"))
                
        kwargs["model"] = self._apply_bitfit(model)
        super().__init__(*args, **kwargs)
        
    def _apply_bitfit(self, model):
        # freeze all parameters except biases
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False
        
        return model
