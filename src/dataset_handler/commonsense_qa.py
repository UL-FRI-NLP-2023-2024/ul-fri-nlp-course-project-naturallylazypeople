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
            dataset["validation"] = dataset["validation"].select(
                range(num_data_points))
            dataset["test"] = dataset["test"].select(range(num_data_points))
            return dataset

    def get_prepcoress_function(self, tokenizer):
        assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

        MAX_LEN = 384
        label_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}

        def preprocess_function(x):
            clean_question = clean_text(x["question"])
            choices = x["choices"]["text"]
            choice_features = []
            for i, choice in enumerate(choices):
                choices[i] = clean_text(choice)

                inputs = tokenizer(
                    clean_question,
                    choices[i],
                    add_special_tokens=True,
                    max_length=MAX_LEN,
                    padding="max_length",
                    truncation=True,
                )

                input_ids = inputs["input_ids"]
                token_type_ids = inputs["token_type_ids"]
                attention_mask = inputs["attention_mask"]

                pad_token_id = tokenizer.pad_token_id
                padding_length = MAX_LEN - len(input_ids)
                input_ids = input_ids + ([pad_token_id] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + \
                    ([pad_token_id] * padding_length)

                assert len(input_ids) == MAX_LEN, "Error with input length {} vs {}".format(
                    len(input_ids), MAX_LEN)
                assert len(attention_mask) == MAX_LEN, "Error with input length {} vs {}".format(
                    len(attention_mask), MAX_LEN)
                assert len(token_type_ids) == MAX_LEN, "Error with input length {} vs {}".format(
                    len(token_type_ids), MAX_LEN)

                choice_features.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                })

            label = label_map.get(x["answerKey"], -1)
            label = torch.tensor(label).long()

            return {
                "id": x["id"],
                "label": label,
                "input_ids": torch.tensor([cf["input_ids"] for cf in choice_features]),
                "attention_mask": torch.tensor([cf["attention_mask"] for cf in choice_features]),
                "token_type_ids": torch.tensor([cf["token_type_ids"] for cf in choice_features]),
            }

        return preprocess_function
