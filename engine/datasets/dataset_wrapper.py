import torch
from torchvision.datasets.folder import default_loader


class DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        item = self.img_list[idx]
        img = self.transform(default_loader(item['img_path']))

        output = {
            "img": img,
            "label": item['label'],
            "classname": item['classname'],
            "img_path": item['img_path']}

        return output
