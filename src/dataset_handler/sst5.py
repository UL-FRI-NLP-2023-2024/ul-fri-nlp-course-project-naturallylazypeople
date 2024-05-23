from dataset_handler.datasets_base import DatasetBase
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import transformers
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast
from utils.utils import clean_text
import stanza
import nltk
from nltk.corpus import stopwords

# Download NLTK resources if not already downloaded
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')

# Download Stanza Slovene model if not already downloaded
# stanza.download("sl")

class SST5Dataset(DatasetBase):
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words('english'))

    def get_model_type(self):
        return AutoModelForSequenceClassification

    def get_dataset(self, num_data_points: int = -1):
        return self.get_dataset_huggingface(num_data_points, "SetFit/sst5")
    
    def get_dataset_task_description(self):
        return "Classify sentiment of the following text:"

    def preprocess_text(self, text, tokenizer):
        # Clean and normalize text
        text = clean_text(text)
        # Tokenize the text into words or subwords (use appropriate tokenizer)
        tokens = tokenizer.tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        # Join tokens back into text
        return ' '.join(tokens)

    def get_preprocess_function(self, tokenizer):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        def preprocess_function(examples):
            # Clean questions and passages (or context)
            cleaned_questions = [self.preprocess_text(q, tokenizer).lstrip()
                                 for q in examples["text"]]

            max_length = 128

            # Tokenize the cleaned inputs
            tokenized_examples = tokenizer(
                cleaned_questions,
                max_length=max_length,
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
                    label = examples["label"][sample_mapping[i]]
                    tokenized_examples["labels"].append(label)

            return tokenized_examples

        return preprocess_function
