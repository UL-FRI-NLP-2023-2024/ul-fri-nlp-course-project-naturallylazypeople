from datasets import load_dataset, Dataset, DatasetDict

class DatasetBase:
    def get_dataset(self, num_data_points):
        raise NotImplementedError

    def get_dataset_task_description(self):
        raise NotImplementedError

    def get_dataset_huggingface(self, num_data_points, huggingface_path, config=None):
        if config is None:
            dataset = load_dataset(huggingface_path)
        else:
            dataset = load_dataset(huggingface_path, config)

        train_df = dataset['train']
        test_df = dataset['test']
        eval_df = dataset['validation']

        if num_data_points != -1:
            train_df = train_df[:num_data_points]
            eval_df = eval_df[:num_data_points]
            test_df = test_df[:num_data_points]

        return DatasetDict({
            'train': Dataset.from_dict(train_df),
            'val': Dataset.from_dict(eval_df),
            'test': Dataset.from_dict(test_df)
        })

    def get_prepcoress_function(self, tokenizer):
        raise NotImplementedError
