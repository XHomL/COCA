import os
from typing import Dict

import numpy as np
import torch
from torch import nn as nn

import engine
import engine.datasets.dataset_wrapper
from engine.clip import partial_model, clip
from engine.model.head import Adapter, LogitHead
from engine.optimizer.default import HYPER_DICT
from engine.optimizer.scheduler import build_lr_scheduler
from engine.templates import get_template
from engine.tools.utils import get_backbone_name, makedirs
from engine.transforms.default import get_image_transforms


def save_text_encoder(clip_model,
                      args):
    text_encoder_dir = get_text_encoder_dir(args)
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")

    if os.path.exists(text_encoder_path):
        print(f"text encoder already saved at {text_encoder_path}")
    else:
        print(f"Saving text encoder to {text_encoder_path}")
        makedirs(text_encoder_dir)
        text_encoder = partial_model.get_text_encoder(args.text_layer_idx, clip_model)
        torch.save(text_encoder, text_encoder_path)


def load_text_encoder(args):
    text_encoder_dir = get_text_encoder_dir(args)
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    text_encoder = torch.load(text_encoder_path)
    return text_encoder


def get_text_encoder_dir(args):
    text_encoder_path = os.path.join(args.feature_dir,
                                     'text',
                                     get_text_encoder_name(args.clip_encoder, args.text_layer_idx))
    return text_encoder_path


def get_text_encoder_name(
        clip_encoder,
        text_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(text_layer_idx)])


def get_image_encoder_name(clip_encoder,
                           image_layer_idx):
    return "_".join([get_backbone_name(clip_encoder), str(image_layer_idx)])


def get_image_encoder_dir(args):
    img_encoder_path = os.path.join(args.feature_dir,
                                    'image',
                                    get_image_encoder_name(args.clip_encoder, args.img_layer_idx))
    return img_encoder_path


def save_image_encoder(
        clip_model,
        args):
    img_encoder_dir = get_image_encoder_dir(args)
    img_encoder_path = os.path.join(img_encoder_dir, "encoder.pth")
    if os.path.exists(img_encoder_path):
        print(f"Image encoder already saved at {img_encoder_path}")
    else:
        print(f"Saving image encoder to {img_encoder_path}")
        makedirs(img_encoder_dir)
        img_encoder = partial_model.get_image_encoder(args.clip_encoder, args.img_layer_idx, clip_model)
        torch.save(img_encoder, img_encoder_path)


def load_image_encoder(args):
    img_encoder_dir = get_image_encoder_dir(args)
    img_encoder_path = os.path.join(img_encoder_dir, "encoder.pth")
    img_encoder = torch.load(img_encoder_path)
    return img_encoder


def extract_image_features(items,
                           img_encoder,
                           args):
    num_views = 1 if args.img_augmentation == 'none' else args.img_views
    assert num_views > 0, "Number of views must be greater than 0"

    train_transform, val_transform = get_image_transforms(args)
    transforms = {'train': train_transform, 'val': val_transform}

    features = {}

    for phase in ['train', 'val']:
        print(f"Extracting image features for {phase} set ...")

        # Setup DataLoader
        loader = torch.utils.data.DataLoader(
            engine.datasets.dataset_wrapper.DatasetWrapper(items[phase], transform=transforms[phase]),
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available())

        img_encoder.feature_extractor.eval()

        # Initialize containers for results
        phase_features, phase_labels, phase_paths = [], [], []

        with torch.no_grad():
            for _ in range(num_views):
                for batch in loader:
                    img = batch["img"].cuda()
                    feature = img_encoder.feature_extractor(img).cpu()

                    # Append results
                    phase_features.append(feature)
                    phase_labels.append(batch['label'])
                    phase_paths.extend(batch['img_path'])

        # Store concatenated features, labels, and paths
        features[phase] = {
            'features': torch.cat(phase_features),
            'labels': torch.cat(phase_labels),
            'paths': phase_paths
        }

    return features


def create_optimizer(classifier, hyperparams):
    params_groups = [{'params': classifier.parameters(), 'lr': hyperparams['lr']}]
    optimizer = torch.optim.AdamW(params_groups, weight_decay=hyperparams['weight_decay'])
    return optimizer


def create_lr_scheduler(optimizer, hyperparams):
    lr_scheduler = build_lr_scheduler(optimizer,
                                   hyperparams['lr_scheduler'],
                                   hyperparams['warmup_iter'],
                                   hyperparams['max_iter'],
                                   warmup_type=hyperparams['warmup_type'],
                                   warmup_lr=hyperparams['warmup_min_lr'])
    return lr_scheduler


def get_hyperparameters(args,
                        img_features,
                        text_features):
    hyperparams = HYPER_DICT[args.hyperparams]
    hyperparams['max_iter'] = 12800

    hyperparams['cross_modal_batch_size'] = 0.5  # Half of the batch is image, the other half is text
    hyperparams['batch_size'] = get_valid_batch_size(img_features,
                                                     text_features,
                                                     hyperparams['cross_modal_batch_size'])
    if args.modality == "cross_modal":
        hyperparams['text_batch_size'] = int(hyperparams['batch_size'] * hyperparams['cross_modal_batch_size'])
    elif args.modality == "uni_modal":
        hyperparams['text_batch_size'] = 0
    hyperparams['image_batch_size'] = hyperparams['batch_size'] - hyperparams['text_batch_size']

    if args.dataset == 'domainnet':
        hyperparams['lr'] = 0.0001
    else:
        hyperparams['lr'] = 0.001
    assert hyperparams['text_batch_size'] <= args.num_classes

    return hyperparams


def get_valid_batch_size(img_features,
                         text_features,
                         batch_ratio):
    candidate_list = []
    # if modality == 'uni_modal':
    #     batch_ratio = 0.
    BATCH_SIZES = [64, 32, 16, 8]
    for batch_size in BATCH_SIZES:
        text_batch_size = int(batch_size * batch_ratio)
        image_batch_size = batch_size - text_batch_size
        # check if text batch size is smaller than the size of text dataset
        if text_batch_size == 0 or text_batch_size < len(text_features):
            # check if image batch size is smaller than the size of image dataset
            if image_batch_size == 0 or image_batch_size < len(img_features):
                candidate_list.append(batch_size)
    if len(candidate_list) == 0:
        raise ValueError("No valid batch size found. You should consider reducing the batch size.")
    valid_batch_size = int(max(np.array(candidate_list)))
    return valid_batch_size


def extract_text_features(texts: Dict[int, str],
                          encoder,
                          args):
    items = {'features': [], 'labels': [], 'eot_indices': []}
    template = get_template(args.text_augmentation)
    encoder.feature_extractor.eval()
    with torch.no_grad():
        for label, cname in texts.items():
            prompt = template.format(cname.replace("_", " "))
            print(f'prompt:{prompt}')

            tokenized_prompt = torch.cat([clip.tokenize(prompt)]).cuda()
            encoder.cuda()

            feature, eot_idx = encoder.feature_extractor(tokenized_prompt)
            items['features'].append(feature.cpu())
            items['labels'].append(torch.tensor(label, dtype=torch.long))
            items['eot_indices'].append(eot_idx.cpu())

    for key in ['features', 'eot_indices']:
        items[key] = torch.cat(items[key])
    items['labels'] = torch.tensor(items['labels'], dtype=torch.long)
    return items


def create_classifier(text_features,
                      args,
                      bias=False):
    num_classes = int(text_features.labels.max()) + 1

    linear_head = nn.Linear(args.feature_dim, num_classes, bias=bias)
    if args.classifier_init == 'zeroshot':
        linear_head.weight.data = load_zero_shot_weights(text_features, num_classes, args.feature_dim)

    if args.classifier_head == 'linear':
        head = linear_head
    elif args.classifier_head == 'adapter':
        adapter = Adapter(args.feature_dim, residual_ratio=0.2)
        head = nn.Sequential(adapter, linear_head)
    else:
        raise ValueError(f"Invalid head: {args.classifier_head}")

    classifier = LogitHead(head, logit_scale=args.logit).train().cuda()
    return classifier


def load_zero_shot_weights(text_features,
                           num_classes,
                           feature_dim):
    with torch.no_grad():
        weights = torch.zeros(num_classes, feature_dim)
        for idx, label in enumerate(text_features.labels):
            weights[label] = text_features.inputs[idx]
        weights.data = torch.nn.functional.normalize(weights, dim=1)
    return weights
