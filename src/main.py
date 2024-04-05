# %%
"""
# Fine-tuning DeBERTA
In this notebook, we will provide the code for fine-tuning DeBERTA.

## Set-up environment

First, we install the libraries which we'll use: HuggingFace Transformers and Datasets.
"""

# %%
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset, load_metric
import numpy as np
import re
import torch

# %%
model_checkpoint = "distilbert-base-uncased"
batch_size = 512

# %%
# clean up to use gpu, ..
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

torch.cuda.set_device(0)

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
testing_only = False
num_data_points = 100
import os

# %%
if data == 'superglue':
    # choose SuperGLUE BoolQ (Yes/No Questions)
    pwd = os.getenv('PWD')

   
    superglue_data_path = os.path.join(pwd, '../data/SuperGLUE-GoogleMT/csv/BoolQ')
    print(superglue_data_path)
    # Load your NLP dataset
    if testing_only:
        train_df = pd.read_csv(f"{superglue_data_path}/train.csv")[:num_data_points]
        eval_df = pd.read_csv(f"{superglue_data_path}/val.csv")[:num_data_points]
    else:
        train_df = pd.read_csv(f"{superglue_data_path}/train.csv")
        eval_df = pd.read_csv(f"{superglue_data_path}/val.csv")


    # Convert data into Hugging Face Dataset format
    dataset_train = Dataset.from_pandas(train_df)
    dataset_eval = Dataset.from_pandas(eval_df)

    # Create a DatasetDict containing the three splits
    dataset = DatasetDict({
        'train': dataset_train,
        'validation': dataset_eval
    })
elif data == 'commensense':
    dataset = load_dataset("commonsense_qa")
elif data == 'xsum':
    dataset = load_dataset("GEM/xsum")

# %%
"""
As we can see, the dataset contains 3 splits: one for training, one for validation and one for testing.
"""

# %%
dataset

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
def clean_text(text):
    # Convert to lowercase
    cleaned_text = text.lower()
    
    # Remove special characters except whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    
    # Remove extra whitespaces
    cleaned_text = ' '.join(cleaned_text.split())
    
    return cleaned_text

# %%
"""
We follow the example of [this](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb#scrollTo=oAeoKVaWaIEl) notebook.
"""

# %%
# Load the pre-trained tokenizer for deberta
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

# %%
pad_on_right = tokenizer.padding_side == "right"
# The maximum length of a feature (question and context)
max_length = 384 
# The authorized overlap between two part of the context when splitting it is needed.
doc_stride = 128 

# %%
def preprocess_function(examples):
    # print(examples.keys()) 
    # Clean questions and passages (or context)
    cleaned_questions = [clean_text(q).lstrip() for q in examples["question"]]
    cleaned_passages = [clean_text(p) for p in examples["passage"]]

    # Tokenize the cleaned inputs
    tokenized_examples = tokenizer(
        cleaned_questions,
        cleaned_passages,
        truncation="only_second",  # Assuming passage comes after question
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["labels"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will use 1 for True and 0 for False
        label = 1 if examples["label"][sample_mapping[i]] == "True" else 0
        tokenized_examples["labels"].append(label)

    return tokenized_examples

# %%
# Pre-process the train, validation, and test datasets
train_dataset = dataset['train'].map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
validation_dataset = dataset['validation'].map(
    preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

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
    evaluation_strategy = "epoch",
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
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    auto_find_batch_size=True,
    num_train_epochs=30,
    weight_decay=0.01,
)

# %%
trainer = Trainer(
    model = model,
    args = args,
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
