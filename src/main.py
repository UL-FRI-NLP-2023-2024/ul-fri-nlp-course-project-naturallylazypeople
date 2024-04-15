############## Parameter Efficient Fine-Tuning ###############

### ------------------ import libraries ------------------ ###
import os
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import torch

from dataset_handler.datasets_base import DatasetBase
from dataset_handler.slo_super_glue import SLOSuperGlueDataset
from dataset_handler.xsum import XSumDataset
from dataset_handler.commonsense_qa import CommonsenseQA

### -------------- configure model and data -------------- ###

#TODO all: change checkpoint
model_checkpoint = "distilbert-base-uncased"
batch_size = 32

# depending on the task, select suitable model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]

# set whether model should be saved
save_model = True

# dataset: choose between 'slo_superglue', 'xsum', 'commensense'
data = 'slo_superglue'
# if you only want to train on subset of data, specify here
num_data_points = 100  # else -1

### --------------------- load dataset --------------------- ###

if data == 'slo_superglue':
    # get path of working directory
    pwd = os.getenv('PWD')
    if pwd is None:
        pwd = os.getcwd()
    superglue_data_path = os.path.join(
        pwd, 'data/SuperGLUE-GoogleMT/csv/')

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

### ----------------- define training args ----------------- ###

args = TrainingArguments(
    output_dir=f"models/{model_name}-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    auto_find_batch_size=True,
    num_train_epochs=30,
    weight_decay=0.01,
)

### ------------------ train the model ------------------- ###

# define HuggingFace's Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

#TODO Ondra: create LoRA Trainer 

# start training the model
trainer.train()

# save model
if save_model:
    trainer.save_model(f"models/{model_name}-{data}-finetuned")

### ----------------- evaluate the model ------------------ ###

#TODO Eileen: add metrics (RAM usage, time, performance metrics, #parameters)
print(trainer.evaluate())

### ---------------- inference on test set ----------------- ###

# TODO: predict on test set
