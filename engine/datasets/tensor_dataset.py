import torch


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor

    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.input_tensor.size(0)
