from dataset_handler.datasets_base import DatasetBase
from datasets import load_dataset, Dataset, DatasetDict
import os
import pandas as pd
from utils.utils import clean_text
import transformers
from transformers import AutoModel


class XSumDataset(DatasetBase):
    def __init__(self) -> None:
        pass

    def get_dataset(self, num_data_points: int = -1):
        return self.get_dataset_huggingface(num_data_points, 'GEM/xsum')

    def get_dataset_task_description(self):
        return "Summarize the following text: "
    
    def get_model_type(self):
        return AutoModel

    def get_prepcoress_function(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        # max_length = 2048  # TODO: find correct values for tokenizer
        doc_stride = 512

        def preprocess_function(examples):
            # Clean questions and passages (or context)
            cleaned_documents = [clean_text(doc).lstrip()
                                 for doc in examples["document"]]
            cleaned_targets = [clean_text(doc).lstrip()
                                 for doc in examples["target"]]

            # Tokenize the cleaned inputs
            tokenized_examples = tokenizer(
                cleaned_documents,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="longest",
            )

            tokenized_targets = tokenizer(
                cleaned_targets,
                # return_overflowing_tokens=True,
                # return_offsets_mapping=True,
                padding="longest",
            )
            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop(
                "overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["labels"] = []

            for i, _ in enumerate(offset_mapping):
                # We will use 1 for True and 0 for False
                # label = examples["target"][sample_mapping[i]]
                # label = {"input_ids": tokenized_targets["input_ids"][sample_mapping[i]],
                #          "token_type_ids": tokenized_targets["token_type_ids"][sample_mapping[i]],
                #          "attention_mask": tokenized_targets["attention_mask"][sample_mapping[i]]}
                label = tokenized_targets["input_ids"][sample_mapping[i]]
                tokenized_examples["labels"].append(label)
            return tokenized_examples

        return preprocess_function
