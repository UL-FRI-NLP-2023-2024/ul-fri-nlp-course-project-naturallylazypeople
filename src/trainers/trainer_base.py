from transformers import Trainer
import numpy as np
import evaluate

class TrainerBase(Trainer):
    def __init__(self, trainer_name, model_name, task_name, model_path, test_dataset, *args, **kwargs):
        super().__init__(*args, compute_metrics=self.compute_metrics, **kwargs)
        self.trainer_name = trainer_name
        self.model_name = model_name
        self.task_name = task_name
        self.model_path = model_path
        self.test_dataset = test_dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        predictions = predictions.flatten()
        labels = labels.flatten()
        logits = logits.flatten()
        
        accuracy = evaluate.load("accuracy").compute(
            predictions=predictions, references=labels)
        f1 = evaluate.load("f1").compute(
            predictions=predictions, references=labels, average="weighted")
        precision = evaluate.load("precision").compute(
            predictions=predictions, references=labels, average="weighted")
        recall = evaluate.load("recall").compute(
            predictions=predictions, references=labels, average="weighted")
        # bleu = evaluate.load("bleu").compute(
        #     predictions=predictions, references=labels)
        # rouge = evaluate.load("rouge").compute(
        #     predictions=predictions, references=labels)

        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision['precision'],
            'f1': f1['f1'],
            'recall': recall['recall'],
            # 'bleu': bleu,
            # 'rouge': rouge,
        }
