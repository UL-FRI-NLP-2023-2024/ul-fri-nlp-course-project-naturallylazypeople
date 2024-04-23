from trainers.trainer_base import TrainerBase
from peft import get_peft_model
from copy import deepcopy

class LoRaTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        model = deepcopy(kwargs.pop("model"))
        lora_config = kwargs.pop("lora_config")
        model = get_peft_model(model, lora_config)

        kwargs["model"] = model
        super().__init__(*args, **kwargs)