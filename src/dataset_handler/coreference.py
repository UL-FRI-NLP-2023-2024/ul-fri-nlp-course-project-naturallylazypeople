from dataset_handler.datasets_base import DatasetBase
import os
import pandas as pd
from datasets import Dataset, DatasetDict
import transformers
from utils.utils import clean_text
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

class CoNLLDataset(DatasetBase):
    def __init__(self, benchmark: str = 'Coreference') -> None:
        self.stop_words = set(stopwords.words('english'))

    def get_dataset(self, num_data_points: int = -1):
        path = 'conll2012_ontonotesv5'
        config = 'english_v12'
        return self.get_dataset_huggingface(num_data_points, path, config)
    
    def get_dataset_task_description(self):
        return "Find all expressions that refer to the same real-world entity"

    def preprocess_text(self, text):
        # Clean and normalize text
        text = clean_text(text)
        # Tokenize the text into words
        words = nltk.word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word.lower() not in self.stop_words]
        # Join tokens back into text
        return ' '.join(words)

    def get_prepcoress_function(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        def preprocess_function(examples):
            # Clean and preprocess the text for both questions and passages
            cleaned_questions = [self.preprocess_text(q) for q in examples["question"]]
            cleaned_passages = [self.preprocess_text(p) for p in examples["passage"]]

            # Tokenize the cleaned inputs
            tokenized_examples = tokenizer(
                cleaned_questions,
                cleaned_passages,
                padding="max_length",
                truncation="only_second",  # Assuming passage comes after question
                return_offsets_mapping=True,
                max_length=512,
            )

            if "label" in examples:
                # Let's label those examples!
                tokenized_examples["labels"] = []
                for i, offset_mapping in enumerate(tokenized_examples["offset_mapping"]):
                    # Get the start and end offsets for the answer in the passage
                    start, end = offset_mapping[examples["start_position"][i]][0], offset_mapping[examples["end_position"][i]][1]
                    # Initialize the label list with 0s
                    labels = [0] * len(offset_mapping)
                    # Set the label for the answer span to 1
                    labels[start:end] = [1] * (end - start)
                    tokenized_examples["labels"].append(labels)

            return tokenized_examples

        return preprocess_function
