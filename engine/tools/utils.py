import json
import os
import random

import numpy as np
import torch


def set_random_seed(seed):
    '''Set random seed for reproducibility.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def makedirs(path, verbose=False):
    '''Make directories if not exist.'''
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if verbose:
            print(path + " already exists.")


def load_json(json_location, default_obj=None):
    '''Load a json file.'''
    if os.path.exists(json_location):
        try:
            with open(json_location, 'r') as f:
                obj = json.load(f)
            return obj
        except:
            print(f"Error loading {json_location}")
            return default_obj
    else:
        return default_obj


def get_view_name(image_augmentation,
                  image_views=1):
    name = f"{image_augmentation}"
    if image_augmentation != "none":
        assert image_views > 0
        name += f"_view_{image_views}"
    return name


def get_lab2cname(args):
    classnames = get_classnames(args)
    container = set()

    for i in range(len(classnames)):
        container.add((i, classnames[i]))

    lab2cname = {label: classname for label, classname in container}
    print(f'lab2cname:{lab2cname}')
    return lab2cname


def get_benchmark_name(
        dataset,
        train_shot,
        seed):
    benchmark_name = "-".join([
        dataset,
        get_few_shot_setup_name(train_shot, seed)
    ])
    return benchmark_name


def get_modality_name(
        modality,
        clip_encoder,
        image_augmentation,
        text_augmentation,
        img_layer_idx,
        text_layer_idx,
        image_views=1):
    text_feature_name = f"text_{text_layer_idx}_{text_augmentation}"
    image_feature_name = f"image_{img_layer_idx}-{get_view_name(image_augmentation, image_views=image_views)}"
    if modality == "cross_modal":
        feature_name = f"{text_feature_name}-{image_feature_name}"
    elif modality == "uni_modal":
        feature_name = image_feature_name

    return os.path.join(get_backbone_name(clip_encoder), feature_name)


def get_architecture_name(
        classifier_head,
        classifier_init):
    return classifier_head + "_" + classifier_init


def get_logit_name(logit):
    name = f"logit_{logit}"
    return name


def get_classifier_dir(args,
                       hyperparams_str):
    classifier_dir = os.path.join(args.record_dir,
                                  args.modality,
                                  args.classifier_head,
                                  get_benchmark_name(args.dataset,
                                                     args.train_shot,
                                                     args.seed),
                                  f'src_{args.source}-num_classes_{args.num_classes}',
                                  get_modality_name(args.modality,
                                                    args.clip_encoder,
                                                    args.img_augmentation,
                                                    args.text_augmentation,
                                                    args.img_layer_idx,
                                                    args.text_layer_idx,
                                                    image_views=args.img_views),
                                  get_architecture_name(args.classifier_head,
                                                        args.classifier_init),
                                  get_logit_name(args.logit),
                                  hyperparams_str)
    return classifier_dir


def get_hyperparams_str(hyperparams):
    hyperparams_str = (f"optim_{hyperparams['optim']}"
                       f"-lr_{hyperparams['lr']}"
                       f"-wd_{hyperparams['weight_decay']}"
                       f"-bs_{hyperparams['batch_size']}")
    return hyperparams_str


def get_few_shot_setup_name(train_shot, seed):
    """Get the name for a few-shot setup.
    """
    return f"shot_{train_shot}-seed_{seed}"


def get_backbone_name(clip_encoder):
    return clip_encoder.replace("/", "-")


def get_classnames(args):
    if args.dataset == 'office31':
        classnames = open(os.path.join(args.img_list_dir,
                                       f'{args.dataset}',
                                       f'{args.source}'
                                       f'/image_unida_list.txt'), 'r').read().splitlines()
        container = set()
        for cls_idx in range(len(classnames)):
            if classnames[cls_idx] != '':
                container.add((int(classnames[cls_idx].split()[1]),
                               classnames[cls_idx].split()[0].split('/')[-2].lower().replace('_', ' ')))
        classnames = []
        for cls_idx in range(31):
            for label, classname in container:
                if label == cls_idx:
                    classnames.append(classname)
    elif args.dataset == 'officehome':
        classnames = ['alarm clock', 'backpack', 'battery', 'bed', 'bike',
                      'bottle', 'bucket', 'calculator', 'calendar', 'candle',
                      'chair', 'clipboard', 'computer', 'couch', 'curtain',
                      'desk lamp', 'drill', 'eraser', 'exit sign', 'fan',
                      'file cabinet', 'flip flops', 'flower', 'folder', 'fork',
                      'glasses', 'hammer', 'helmet', 'kettle', 'keyboard',
                      'knife', 'lamp shade', 'laptop', 'marker', 'monitor',
                      'mop', 'mouse', 'mug', 'notebook', 'oven',
                      'pan', 'paper clip', 'pen', 'pencil', 'post-it note',
                      'printer', 'push pin', 'radio', 'refrigerator', 'ruler',
                      'scissor', 'screwdriver', 'shelf', 'sink', 'sneakers',
                      'soda', 'speaker', 'spoon', 'table', 'telephone',
                      'toothbrush', 'toy', 'trash can', 'TV', 'webcam']
    elif args.dataset == 'visda':
        classnames = ['aeroplane', 'bicycle', 'bus', 'car', 'horse',
                      'knife', 'motorcycle', 'person', 'plant', 'skateboard',
                      'train', 'truck']
    elif args.dataset == 'domainnet':
        classnames = open(os.path.join(args.img_list_dir,
                                       f'{args.dataset}',
                                       f'{args.source}'
                                       f'/image_unida_list.txt'), 'r').read().split("\n")
        container = set()
        for cls_idx in range(len(classnames)):
            if classnames[cls_idx] != '':
                container.add((int(classnames[cls_idx].split()[1]),
                               classnames[cls_idx].split()[0].split('/')[-2].lower().replace('_', ' ')))
        classnames = []
        for cls_idx in range(345):
            for label, classname in container:
                if label == cls_idx:
                    classnames.append(classname)
    return classnames[:args.num_classes]
