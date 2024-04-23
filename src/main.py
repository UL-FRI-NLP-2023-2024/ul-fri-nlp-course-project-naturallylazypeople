############## Parameter Efficient Fine-Tuning ###############

### ------------------ import libraries ------------------ ###
import os

from dataset_handler.datasets_base import DatasetBase
from dataset_handler.slo_super_glue import SLOSuperGlueDataset
from dataset_handler.xsum import XSumDataset
from dataset_handler.commonsense_qa import CommonsenseQA

from evaluator.evaluator_base import EvaluatorBase

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, TrainingArguments
from trainers.trainer_base import TrainerBase
from trainers.trainer_lora import LoRaTrainer

from peft import LoraConfig

import torch

### -------------- configure model and data -------------- ###

#TODO all: change checkpoint  # microsoft/deberta-v2  # microsoft/deberta-v2-xlarge
model_checkpoint = "distilbert-base-uncased"
batch_size = 32

# depending on the task, select suitable model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
model_path = f"output/models/{model_name}"

# set whether model should be saved
save_model = True

# dataset: choose between 'slo_superglue', 'xsum', 'commensense'
data = 'slo_superglue'
# if you only want to train on subset of data, specify here
num_data_points = 20  # else -1

### --------------------- load dataset --------------------- ###

if data == 'slo_superglue':
    # get path of working directory
    pwd = os.getenv('PWD')
    if pwd is None:
        pwd = os.getcwd()
    superglue_data_path = os.path.join(
        pwd, 'data/SuperGLUE-GoogleMT/csv/')
    superglue_data_path = '/mnt/c/Users/komin/ownCloud - Bc. Ondřej Komín@owncloud.cesnet.cz/magistr/4. semestr/NLP/Project/ul-fri-nlp-course-project-naturallylazypeople/data/SuperGLUE-GoogleMT/csv/'

    dataset: DatasetBase = SLOSuperGlueDataset(
        superglue_data_path, 'BoolQ')
elif data == 'commensense':
    dataset: DatasetBase = XSumDataset()
elif data == 'xsum':
    dataset: DatasetBase = CommonsenseQA()

else:
    raise RuntimeError(f"Dataset {data} is not supported")


dataset_data = dataset.get_dataset(num_data_points)

### ------------- load pre-trained tokenizer ------------- ###

# Load the pre-trained tokenizer for deberta
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# load preprocess function
preprocess_function = dataset.get_prepcoress_function(tokenizer)

### ------------------ pre-process data ------------------ ###

# train dataset
train_dataset = dataset_data['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_data["train"].column_names)
# validation dataset
val_dataset = dataset_data['val'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_data["train"].column_names)

# set format of data to PyTorch tensors
train_dataset.set_format("torch")
val_dataset.set_format("torch")

### -------------- define training arguments -------------- ###

# we might want to give different arguments to different trainers in the future
args = TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    auto_find_batch_size=True,
    num_train_epochs=10,
    weight_decay=0.01,
)

### ------------------- define trainers ------------------- ###

# full fine-tuning trainer
ft_path = f"output/models/{model_name}-{data}-fft"
trainer_ft = TrainerBase(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    trainer_name='fft',
    model_name=model_name,
    task_name=data,
    model_path=ft_path,
)

# LoRA trainer
lora_path = f"output/models/{model_name}-{data}-lora"
lora_config = LoraConfig(  # so far hardcoded
        r=16,
        lora_alpha=32,  # rule of thumb alpha = 2*r
        target_modules=["q_lin", "k_lin","v_lin"],  # The modules (for example, attention blocks) to apply the LoRA update matrices.
        lora_dropout=0.1,
        bias="lora_only",
        modules_to_save=None,  # List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. These typically include model’s custom head that is randomly initialized for the fine-tuning task.
        task_type="SEQ_CLS"
    )

trainer_lora = LoRaTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    trainer_name='lora',
    model_name=model_name,
    task_name=data,
    model_path=lora_path,
    lora_config=lora_config
)

# create list of all trainers that we want to compare against each other
trainers = [trainer_ft, trainer_lora]
### ------------ train and evaluate the model ------------- ###

# train and evaluate all trainers by passing to Evaluator
evaluator = EvaluatorBase(trainers)
evaluator.train_and_evaluate(save_model)

# get metrics
metrics = evaluator.get_metrics()
evaluator.save_metrics(f"output/metrics/metrics.json")

### --------------- inference on test set ----------------- ###

# TODO: predict on test set

print('Done')