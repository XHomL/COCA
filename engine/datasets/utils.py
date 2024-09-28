import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

from engine.datasets.tensor_dataset import TensorDataset
from engine.datasets.text_tensor_dataset import TextTensorDataset
from engine.model.utils import (get_image_encoder_dir,
                                get_text_encoder_dir)
from engine.tools.utils import (get_view_name,
                                get_few_shot_setup_name,
                                makedirs,
                                load_json)

VALID_DOMAINS = {
    'office31': ['amazon', 'dslr', 'webcam'],
    'officehome': ['art', 'clipart', 'product', 'realworld'],
    'visda': ['train', 'validation'],
    'domainnet': ['painting', 'real', 'sketch']
}


def validate_domain(dataset: str, domain: str):
    assert domain in VALID_DOMAINS[dataset], f"Invalid domain '{domain}' for dataset '{dataset}'"


def get_text_features_path(args):
    text_features_path = os.path.join(get_text_encoder_dir(args),
                                      args.dataset,
                                      f'num_classes_{args.num_classes}',
                                      f"{args.text_augmentation}.pth")
    return text_features_path


def load_text_features(args):
    text_features_path = get_text_features_path(args)
    text_features = torch.load(text_features_path)
    text_features = TextTensorDataset(text_features['features'],
                                     text_features['labels'],
                                     text_features['eot_indices'])
    return text_features


def save_text_features(features,
                       args):
    features_path = get_text_features_path(args)

    if os.path.exists(features_path):
        print(f"Text features already saved at {features_path}")
    else:
        print(f"Saving text features to {features_path}")
        makedirs(os.path.dirname(features_path))
        torch.save(features, features_path)


def get_image_features_path(args,
                            image_augmentation: Optional[str] = None):
    augmentation = image_augmentation if image_augmentation is not None else args.img_augmentation

    image_features_path = os.path.join(get_image_encoder_dir(args),
                                       args.dataset,
                                       f'src_{args.source}-num_classes_{args.num_classes}',
                                       get_view_name(augmentation, args.img_views),
                                       f'{get_few_shot_setup_name(args.train_shot, args.seed)}.pth')
    return image_features_path


def load_image_features(args):
    ccrop_features_path = get_image_features_path(args, 'none')
    ccrop_features = torch.load(ccrop_features_path)
    if args.img_augmentation == "none":
        train_features = ccrop_features['train']['features']
        train_labels = ccrop_features['train']['labels']
    else:
        # Add extra views
        features_path = get_image_features_path(args)
        features = torch.load(features_path)
        train_features = torch.cat([ccrop_features['train']['features'], features['train']['features']],
                                   dim=0)
        train_labels = torch.cat([ccrop_features['train']['labels'], features['train']['labels']],
                                 dim=0)
    train_features = TensorDataset(train_features,
                                   train_labels)
    val_features = TensorDataset(ccrop_features['val']['features'],
                                ccrop_features['val']['labels'])
    return train_features, val_features


def save_image_features(features,
                        args):
    features_path = get_image_features_path(args)
    if os.path.exists(features_path):
        print(f"Image features already saved at {features_path}")
    else:
        print(f"Saving image features to {features_path}")
        makedirs(os.path.dirname(features_path))
        torch.save(features, features_path)


def get_train_loaders(img_features,
                      text_features,
                      args,
                      hyperparams):
    img_features_loader = None
    if hyperparams['image_batch_size'] > 0:
        img_features_loader = DataLoader(img_features,
                                         batch_size=hyperparams['image_batch_size'],
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=True,
                                         persistent_workers=True)
    text_features_loader = None
    if hyperparams['text_batch_size'] > 0:
        text_features_loader = DataLoader(text_features,
                                          batch_size=hyperparams['text_batch_size'],
                                          shuffle=True,
                                          num_workers=args.num_workers,
                                          pin_memory=True,
                                          drop_last=True,
                                          persistent_workers=True)
    return img_features_loader, text_features_loader


def get_val_loader(features,
                   args,
                   hyperparams):
    features_loader = DataLoader(features,
                                 batch_size=hyperparams['batch_size'],
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 persistent_workers=True)
    return features_loader


def load_few_shot_items(args):
    items_file = os.path.join(args.few_shot_items_dir,
                              args.dataset,
                              f'src_{args.source}-num_classes_{args.num_classes}',
                              f'{get_few_shot_setup_name(args.train_shot, args.seed)}.json')
    assert os.path.exists(items_file), f"Few-shot data does not exist at {items_file}."
    few_shot_items = load_json(items_file)
    return {'train': few_shot_items['train']['img_list'],
            'val': few_shot_items['val']['img_list']}
