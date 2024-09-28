import torch


class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, eot_indices):
        self.inputs = inputs
        self.labels = labels
        self.eot_indices = eot_indices

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index], self.eot_indices[index]

    def __len__(self):
        return self.inputs.size(0)
