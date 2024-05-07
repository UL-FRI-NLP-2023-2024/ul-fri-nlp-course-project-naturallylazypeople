from trainers.trainer_base import TrainerBase
from peft import get_peft_model, LoraConfig
from copy import deepcopy

class LoRaTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        model = deepcopy(kwargs.pop("model"))
        task_type = kwargs.pop("task_type")

        lora_config = LoraConfig(  # so far hardcoded
            r=16,
            lora_alpha=32,  # rule of thumb alpha = 2*r
            # target_modules=["q_lin", "k_lin", "v_lin"],  
            lora_dropout=0.1,
            bias="lora_only",
            modules_to_save=None, 
            task_type=task_type,
        )
        model = get_peft_model(model, lora_config)

        kwargs["model"] = model
        super().__init__(*args, **kwargs)