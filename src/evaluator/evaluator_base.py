import psutil
import time
import json
from utils import utils
import torch

class EvaluatorBase:
    def __init__(self, trainers):
        self.trainers = trainers
        self.metrics = {}

    def compute_ram_usage(self):
        process = psutil.Process()
        ram_usage = process.memory_info().rss / 1024 ** 2  # in MB
        return ram_usage
    
    def compute_cpu_usage(self):
        cpu_usage = psutil.cpu_percent(interval=1)
        return cpu_usage
    
    def compute_gpu_memory_usage(self):
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        return memory_allocated, memory_reserved

    def compute_training_time(self, start_time):
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def train_and_evaluate(self, save_model):

        trainers = []
        for trainer in self.trainers:
            start_time = time.time()
            trainer.train()
            trainers.append(trainer)
            training_time = self.compute_training_time(start_time)

            metric = {}
            
            metric['task_name'] = trainer.task_name
            metric['model_name'] = trainer.model_name
            metric['model_path'] = trainer.model_path

            metric["training_time"] = training_time
            metric["ram_usage"] = self.compute_ram_usage()
            metric["cpu_usage"] = self.compute_cpu_usage()
            metric["gpu_usage"] = self.compute_gpu_memory_usage()
            metric["all_params"] = utils.trainable_parameters(trainer.model, print=False)["all_params"]
            metric["trainable_params"] = utils.trainable_parameters(trainer.model, print=False)["trainable_params"]

            # Add evaluation metrics from Hugging Face's evaluate module
            eval_result_metric = trainer.evaluate()
            metric.update(eval_result_metric)

            self.metrics[trainer.trainer_name] = metric

            if save_model:
                trainer.save_model(trainer.model_path)

        return trainers

    def get_metrics(self):
        return self.metrics
    
    def save_metrics(self, output_file):
        
        metrics = self.metrics
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=4) 

    def inference_on_test_set(self, trainer, model_type):
        loaded_model = model_type.from_pretrained(trainer.model_path)
        trainer.model = loaded_model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_model.to(device)
        test_dataset_device = trainer.test_dataset.map(
            lambda x: {key: value.to(device) for key, value in x.items()})
        predictions = trainer.predict(test_dataset_device)
        return predictions
