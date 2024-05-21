from dataset_handler.datasets_base import DatasetBase
from datasets import load_dataset, DatasetDict
from utils.utils import clean_text
import transformers
import torch


class CommonsenseQA(DatasetBase):
    def __init__(self) -> None:
        pass

    def get_model_type(self):
        return transformers.AutoModelForMultipleChoice 

    def get_dataset_task_description(self):
        return "Choose the correct answer to the question."

    def get_dataset(self, num_data_points: int = -1):
        if num_data_points == -1:
            return load_dataset("commonsense_qa")
        else:
            dataset = load_dataset("commonsense_qa")
            dataset["train"] = dataset["train"].select(range(num_data_points))
            dataset["validation"] = dataset["validation"].select(range(num_data_points))
            dataset["test"] = dataset["test"].select(range(num_data_points))
            return dataset

    def get_preprocess_function(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        MAX_LEN = 384
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        def preprocess_function(batch):
            clean_questions = [clean_text(q) for q in batch["question"]]
            all_choices = [choices["text"] for choices in batch["choices"]]

            input_ids = []
            attention_masks = []
            token_type_ids = []

            for question, choices in zip(clean_questions, all_choices):
                clean_choices = [clean_text(choice) for choice in choices]
                inputs = tokenizer(
                    [question] * len(clean_choices),
                    clean_choices,
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids.append(inputs["input_ids"])
                attention_masks.append(inputs["attention_mask"])
                token_type_ids.append(inputs["token_type_ids"])

            # Stack the inputs for all choices in each question
            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)
            token_type_ids = torch.stack(token_type_ids)

            labels = [label_map.get(answer, -1) for answer in batch["answerKey"]]

            return {
                "id": batch["id"],
                "label": torch.tensor(labels, dtype=torch.long),
                "input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids,
            }

        return preprocess_function
