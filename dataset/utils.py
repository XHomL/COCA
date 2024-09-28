import os

import torch
from torch.utils.data import DataLoader

from dataset.dataset import SFUniDADataset
from engine.datasets.text_tensor_dataset import TextTensorDataset
from engine.datasets.utils import get_text_features_path


def create_img_loaders(args):
    target_img_list = open(os.path.join(args.target_dir, "image_unida_list.txt"), "r").readlines()
    target_img_dataset = SFUniDADataset(
        args,
        args.target_dir.replace('./image_lists', '/mnt/data_ssd/xhliu/data'),
        target_img_list,
        domain_type="target",
        preload_flg=args.preload_flag)
    img_loader_train = DataLoader(
        target_img_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True)
    img_loader_test = DataLoader(
        target_img_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False)
    return img_loader_train, img_loader_test


def create_text_features_loader(args):
    text_features_path = get_text_features_path(args)
    text_features = torch.load(text_features_path)
    text_features = TextTensorDataset(text_features['features'],
                                      text_features['labels'],
                                      text_features['eot_indices'])
    args.text_batch_size = min(args.batch_size, args.num_classes)
    text_features_loader = DataLoader(text_features,
                                      batch_size=args.text_batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True,
                                      drop_last=True,
                                      persistent_workers=True)
    return text_features_loader
