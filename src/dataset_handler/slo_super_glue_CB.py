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
stanza.download("sl")

class SLOSuperGlueCBDataset(DatasetBase):
    def __init__(self, path: str, benchmark: str = 'CB') -> None:
        self.path = os.path.join(path, benchmark)
        self.stop_words = set(stopwords.words('slovene'))
        self.nlp = stanza.Pipeline("sl")

    def get_model_type(self):
        return AutoModelForSequenceClassification

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
    
    def get_dataset_task_description(self):
        return "Ali je naslednja posledica, protislovje ali nevtralno?: "

    def preprocess_text(self, text):
        # Clean and normalize text
        text = clean_text(text)
        # Lemmatize text using Stanza
        doc = self.nlp(text)
        lemmatized_tokens = [word.lemma for sentence in doc.sentences for word in sentence.words]
        # Remove stopwords
        lemmatized_tokens = [token for token in lemmatized_tokens if token.lower() not in self.stop_words]
        # Join tokens back into text
        return ' '.join(lemmatized_tokens)

    def get_preprocess_function(self, tokenizer):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        max_length = 500
        # doc_stride = 128
        label_map = {"entailment": 0, "contradiction": 1, "neutral": 2}

        def preprocess_function(examples):
            # Clean questions and passages (or context)
            cleaned_premises = [self.preprocess_text(q).lstrip()
                                 for q in examples["premise"]]
            cleaned_hypothesis = [self.preprocess_text(p) for p in examples["hypothesis"]]

            # Tokenize the cleaned inputs
            tokenized_examples = tokenizer(
                cleaned_premises,
                cleaned_hypothesis,
                # truncation="only_second",  # Assuming passage comes after question
                max_length=max_length,
                # stride=doc_stride,
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
                    label = label_map[label]
                    tokenized_examples["labels"].append(label)

            return tokenized_examples

        return preprocess_function
