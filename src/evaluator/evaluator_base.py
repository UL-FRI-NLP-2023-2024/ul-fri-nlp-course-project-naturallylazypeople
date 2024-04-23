import psutil
import time
import json
from utils import utils

class EvaluatorBase:
    def __init__(self, trainers):
        self.trainers = trainers
        self.metrics = {}

    def compute_ram_usage(self):
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024 ** 2  # in MB
        return ram_usage

    def compute_training_time(self, start_time):
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def train_and_evaluate(self, save_model):

        for trainer in self.trainers:
            start_time = time.time()
            trainer.train()
            training_time = self.compute_training_time(start_time)

            metric = {}
            
            metric['task_name'] = trainer.task_name
            metric['model_name'] = trainer.model_name
            metric['model_path'] = trainer.model_path

            metric["training_time"] = training_time
            metric["ram_usage"] = self.compute_ram_usage()
            metric["all_params"] = utils.trainable_parameters(trainer.model, print=False)["all_params"]
            metric["trainable_params"] = utils.trainable_parameters(trainer.model, print=False)["trainable_params"]

            # Add evaluation metrics from Hugging Face's evaluate module
            metric_names = ["accuracy", "precision", "f1", "bleu", "rouge"]
            for metric_name in metric_names:
                eval_result_metric = trainer.evaluate(metric_key_prefix=metric_name)
                metric.update(eval_result_metric)

            self.metrics[trainer.trainer_name] = metric

            if save_model:
                trainer.save_model(trainer.model_path)

    def get_metrics(self):
        return self.metrics
    
    def save_metrics(self, output_file):
        
        metrics = self.metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4) 
                
    def eval_dataset():
        # TODO: eval testset
        pass