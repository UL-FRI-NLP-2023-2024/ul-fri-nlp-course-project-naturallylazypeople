from transformers import Trainer

class TrainerBase(Trainer):
    def __init__(self, trainer_name, model_name, task_name, model_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainer_name = trainer_name
        self.model_name = model_name
        self.task_name = task_name
        self.model_path = model_path