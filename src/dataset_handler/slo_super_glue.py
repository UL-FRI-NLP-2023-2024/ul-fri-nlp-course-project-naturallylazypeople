from dataset_handler.datasets_base import DatasetBase
from datasets import Dataset, DatasetDict
import os
import pandas as pd
from utils.utils import clean_text
import transformers

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Download NLTK resources if not already downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

class SLOSuperGlueDataset(DatasetBase):
    def __init__(self, path: str, benchmark: str = 'BoolQ') -> None:
        self.path = os.path.join(path, benchmark)
        self.stop_words = set(stopwords.words('slovene'))
        self.stemmer = SnowballStemmer('slovene')

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
            'val': Dataset.from_pandas(eval_df),
            'test': Dataset.from_pandas(test_df)
        })

    def preprocess_text(self, text):
        # Clean and normalize text
        text = clean_text(text)
        # Tokenize text
        tokens = word_tokenize(text, language='slovene')
        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        # Stemming
        tokens = [self.stemmer.stem(token) for token in tokens]
        # Join tokens back into text
        return ' '.join(tokens)

    def get_prepcoress_function(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        max_length = 384
        doc_stride = 128

        def preprocess_function(examples):
            # Clean questions and passages (or context)
            cleaned_questions = [self.preprocess_text(q).lstrip()
                                 for q in examples["question"]]
            cleaned_passages = [self.preprocess_text(p) for p in examples["passage"]]

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

            if "label" in examples:
                # Let's label those examples!
                tokenized_examples["labels"] = []
                for i, _ in enumerate(offset_mapping):
                    # We will use 1 for True and 0 for False
                    if examples["label"][sample_mapping[i]]:
                        label = 1 
                    else:
                        label = 0
                    tokenized_examples["labels"].append(label)
            # else:
            #     tokenized_examples["labels"] = [0] * len(offset_mapping)  # Placeholder labels

            return tokenized_examples

        return preprocess_function