class DatasetBase:
    def get_dataset(self, num_data_points):
        raise NotImplementedError

    def get_prepcoress_function(self, tokenizer):
        raise NotImplementedError
