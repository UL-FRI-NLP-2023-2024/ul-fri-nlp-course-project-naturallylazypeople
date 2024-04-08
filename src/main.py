# %%
"""
# Fine-tuning DeBERTA
In this notebook, we will provide the code for fine-tuning DeBERTA.

## Set-up environment

First, we install the libraries which we'll use: HuggingFace Transformers and Datasets.
"""

# %%
import os
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_metric
import numpy as np
import re
import torch

from Dataset import DatasetBase
from SLOSuperGlueDataset import SLOSuperGlueDataset

# %%
model_checkpoint = "distilbert-base-uncased"
batch_size = 32

if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda")
    torch.cuda.set_device(0)
elif torch.backends.mps.is_available():
    print("Using MPS")
    device = torch.device("mps")
else:
    print("Using CPU")
    device = torch.device("cpu")


# %%
"""
## Load dataset

We will read the three csv files (train, test, validation) and convert them to a HuggingFace Dataset format.
"""

# %%
# choose between 'superglue', 'xsum', 'commensense'
data = 'superglue'

# %%
#### CHOOSE SIZE OF DATA ####
# if you just want to try out the code, select a small number
testing_only = True
num_data_points = 100

# %%
if data == 'superglue':
    pwd = os.getenv('PWD')
    if pwd is None:
        pwd = os.getcwd()

    superglue_data_path = os.path.join(
        pwd, '../data/SuperGLUE-GoogleMT/csv/')
    dataset: DatasetBase = SLOSuperGlueDataset(
        superglue_data_path, 'BoolQ')
elif data == 'commensense':
    dataset = load_dataset("commonsense_qa")
elif data == 'xsum':
    dataset = load_dataset("GEM/xsum")

# %%
"""
As we can see, the dataset contains 3 splits: one for training, one for validation and one for testing.
"""

# %%
dataset_data = dataset.get_dataset(num_data_points)
# %%
"""
Let's check the first example of the training split:
"""

# %%
# print(dataset.keys())
# example = dataset['train'][0]
# example

# %%
"""
## Preprocess data

As models like BERT don't expect text as direct input, but rather `input_ids`, etc., we tokenize the text using the tokenizer. Here I'm using the `AutoTokenizer` API, which will automatically load the appropriate tokenizer based on the checkpoint on the hub.
"""

# %%
# Load the pre-trained tokenizer for deberta
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


preprocess_function = dataset.get_prepcoress_function(tokenizer)


# %%
# Pre-process the train, validation, and test datasets
train_dataset = dataset_data['train'].map(
    preprocess_function, batched=True, remove_columns=dataset_data["train"].column_names)
validation_dataset = dataset_data['validation'].map(
    preprocess_function, batched=True, remove_columns=dataset_data
    ["train"].column_names)


# %%
example = train_dataset[0]
print(example.keys())

# %%
tokenizer.decode(example['input_ids'])

# %%
"""
Finally, we set the format of our data to PyTorch tensors. This will turn the training, validation and test sets into standard PyTorch [datasets](https://pytorch.org/docs/stable/data.html).
"""

# %%
train_dataset.set_format("torch")

# %%
"""
## Define model

Here we define a model that includes a pre-trained base (i.e. the weights from bert-base-uncased) are loaded, with a random initialized classification head (linear layer) on top. One should fine-tune this head, together with the pre-trained base on a labeled dataset.

This is also printed by the warning.

We set the `problem_type` to be "multi_label_classification", as this will make sure the appropriate loss function is used (namely [`BCEWithLogitsLoss`](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)). We also make sure the output layer has `len(labels)` output neurons, and we set the id2label and label2id mappings.
"""

# %%
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-squad",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=300,
    weight_decay=0.01,
    push_to_hub=True,
)

print(args.device)

# %%
"""
## Train the model!

We are going to train the model using HuggingFace's Trainer API. This requires us to define 2 things:

* `TrainingArguments`, which specify training hyperparameters. All options can be found in the [docs](https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments). Below, we for example specify that we want to evaluate after every epoch of training, we would like to save the model every epoch, we set the learning rate, the batch size to use for training/evaluation, how many epochs to train for, and so on.
* a `Trainer` object (docs can be found [here](https://huggingface.co/transformers/main_classes/trainer.html#id1)).
"""

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

# %%
model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-boolq",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    auto_find_batch_size=True,
    num_train_epochs=30,
    weight_decay=0.01,
)

# %%
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

# %%
"""
Let's start training!
"""

# %%
trainer.train()

# %%
save_model = True

if save_model:
    trainer.save_model(f"{model_name}-finetuned-boolq")

# %%
"""
## Evaluate

After training, we evaluate our model on the validation set.
"""

# %%
trainer.evaluate()

# %%
"""
## Inference

Let's test the model on a new sentence:
"""

# %%
