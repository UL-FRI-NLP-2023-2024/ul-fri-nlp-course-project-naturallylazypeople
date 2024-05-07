from trainers.trainer_base import TrainerBase
from peft import get_peft_model, IA3Config
from copy import deepcopy

class IA3Trainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        model = deepcopy(kwargs.pop("model"))
        
        task_type = kwargs.pop("task_type")

        ia3config = IA3Config(
            task_type=task_type
        )
        model = get_peft_model(model, ia3config)

        kwargs["model"] = model
        super().__init__(*args, **kwargs)