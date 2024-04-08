from Dataset import DatasetBase
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from utils import clean_text
import transformers


class SLOSuperGlueDataset(DatasetBase):
    def __init__(self, path: str, benchmark: str = 'BoolQ') -> None:
        self.path = os.path.join(path, benchmark)

    def get_dataset(self, num_data_points: int = -1):
        train_df = pd.read_csv(f"{self.path}/train.csv")
        eval_df = pd.read_csv(f"{self.path}/val.csv")
        test_df = pd.read_csv(f"{self.path}/test.csv")

        if num_data_points != -1:
            train_df = train_df[:num_data_points]
            eval_df = eval_df[:num_data_points]
            test_df = test_df[:num_data_points]

        return DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(eval_df),
            'test': Dataset.from_pandas(test_df)
        })

    def get_prepcoress_function(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        max_length = 384
        doc_stride = 128

        def preprocess_function(examples):
            # print(examples.keys())
            # Clean questions and passages (or context)
            cleaned_questions = [clean_text(q).lstrip()
                                 for q in examples["question"]]
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
            sample_mapping = tokenized_examples.pop(
                "overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["labels"] = []

            for i, _ in enumerate(offset_mapping):
                # We will use 1 for True and 0 for False
                label = 1 if examples["label"][sample_mapping[i]
                                               ] == "True" else 0
                tokenized_examples["labels"].append(label)

            return tokenized_examples

        return preprocess_function
