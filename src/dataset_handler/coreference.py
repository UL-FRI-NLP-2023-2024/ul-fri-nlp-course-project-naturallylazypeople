from dataset_handler.datasets_base import DatasetBase
from transformers import AutoModelForTokenClassification, PreTrainedTokenizerFast
from utils.utils import clean_text
import nltk
from nltk.corpus import stopwords
import torch
from collections import defaultdict
import datasets
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

class CoNLLDataset(DatasetBase):
    def __init__(self, benchmark: str = 'Coreference') -> None:
        self.stop_words = set(stopwords.words('arabic'))

    def get_model_type(self):
        return AutoModelForTokenClassification

    def get_dataset(self, num_data_points: int = -1):
        path = 'conll2012_ontonotesv5'
        config = 'arabic_v4'
        return self.get_dataset_huggingface(num_data_points, path, config)
    
    def get_dataset_task_description(self):
        return "Find all expressions that refer to the same real-world entity"

    def preprocess_text(self, text, tokenizer):
        # Clean and normalize text
        text = clean_text(text)
        # Tokenize the text into words or subwords (use appropriate tokenizer)
        tokens = tokenizer.tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        # Join tokens back into text
        return ' '.join(tokens)

    def get_prepcoress_function(self, tokenizer: PreTrainedTokenizerFast):
        assert isinstance(tokenizer, PreTrainedTokenizerFast)

        max_length = 384

        def preprocess_function(documents, tokenizer=tokenizer):
            all_tokenized_sentences = defaultdict(list)
            for doc in documents['sentences']:
                coref_chains = self.extract_coref_chains(doc)
                tokenized_sentences = self.tokenize_and_align_labels(
                    doc, coref_chains, tokenizer, max_length=max_length)
                for key, value in tokenized_sentences.items():
                    all_tokenized_sentences[key].extend(value)

            # Convert lists of lists to lists of tensors for the final dataset
            for key, value in all_tokenized_sentences.items():
                all_tokenized_sentences[key] = torch.tensor(value)

            return all_tokenized_sentences
        
        return preprocess_function

    def extract_coref_chains(self, sentences):
        coref_chains = []
        for sentence in sentences:
            coref_spans = sentence.get('coref_spans', [])
            chains = defaultdict(list)
            for span in coref_spans:
                cluster_id, start_idx, end_idx = span
                chains[cluster_id].append((start_idx, end_idx))
            coref_chains.append(chains)
        return coref_chains
    
    def tokenize_and_align_labels(self, sentences, coref_chains, tokenizer, max_length=128):
        tokenized_sentences = defaultdict(list)
        for sentence, chains in zip(sentences, coref_chains):
            words = sentence['words']
            labels = [0] * len(words)  # Initialize with 0 (no coreference)
            
            for chain_id, spans in chains.items():
                for (start_idx, end_idx) in spans:
                    for i in range(start_idx, end_idx + 1):
                        labels[i] = chain_id  # Set label to the cluster ID
            
            tokenized_inputs = tokenizer(
                words, 
                is_split_into_words=True, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt', 
                max_length=max_length)
            
            # Align labels with tokenized inputs
            word_ids = tokenized_inputs.word_ids()
            aligned_labels = []
            current_word = None
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(0)  # Special tokens are ignored in the loss function
                elif word_id != current_word:
                    aligned_labels.append(labels[word_id])
                    current_word = word_id
                else:
                    aligned_labels.append(labels[word_id])
            
            # Convert aligned labels to int32 and pad to max_length
            aligned_labels = np.array(aligned_labels, dtype=np.int32)
            padding_length = tokenized_inputs['input_ids'].shape[1] - len(aligned_labels)
            aligned_labels = np.pad(aligned_labels, (0, padding_length), mode='constant', constant_values=-100)
            
            tokenized_inputs['labels'] = torch.tensor(aligned_labels, dtype=torch.int32).unsqueeze(0)  # Convert to tensor and add batch dimension

            # Separate each tensor in tokenized_inputs
            for key, value in tokenized_inputs.items():
                tokenized_sentences[key].append(value.squeeze(0).tolist())
        
        return tokenized_sentences

    def convert_to_hf_dataset(self, tokenized_inputs):
        # Create HuggingFace dataset
        hf_dataset = datasets.Dataset.from_dict(tokenized_inputs)
        return hf_dataset
