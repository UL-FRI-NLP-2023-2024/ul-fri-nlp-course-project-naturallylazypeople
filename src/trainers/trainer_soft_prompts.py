from trainers.trainer_base import TrainerBase
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig
from copy import deepcopy

class SoftPromptsTrainer(TrainerBase):
    def __init__(self, *args, **kwargs):
        model = deepcopy(kwargs.pop("model"))
        
        tokenizer = kwargs.pop("tokenizer")
        initial_text = kwargs.pop("initial_text")
        num_tokens = kwargs.pop("num_tokens") if "num_tokens" in kwargs.keys() else 8
        task_type = kwargs.pop("task_type")

        soft_prompts_config = PromptTuningConfig(
            task_type=task_type,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=num_tokens,
            prompt_tuning_init_text=initial_text,
            tokenizer_name_or_path=tokenizer,
        ) 

        model = get_peft_model(model, soft_prompts_config)

        kwargs["model"] = model
        super().__init__(*args, **kwargs)