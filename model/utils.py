import os

import torch

from engine.datasets.utils import load_text_features, load_image_features
from engine.model.utils import get_image_encoder_dir, get_text_encoder_dir, get_hyperparameters
from engine.tools.utils import get_hyperparams_str, get_classifier_dir
from model.masking import Masking
from model.sfunida_clip import SFUniDACLIP
from utils.net_utils import duplicate_init_lr


def create_masking(args):
    if args.clip_encoder in ['RN50x16', 'RN50']:
        masking = Masking(
            block_size=4,
            ratio=args.mask_ratio,
            color_jitter_s=0.,
            color_jitter_p=0.,
            blur=False,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
    else:
        masking = Masking(
            block_size=16,
            ratio=args.mask_ratio,
            color_jitter_s=0.2,
            color_jitter_p=0.2,
            blur=True,
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711))
    return masking


def load_source_model(args):
    img_encoder_dir = get_image_encoder_dir(args)
    img_encoder_path = os.path.join(img_encoder_dir, "encoder.pth")
    text_features = load_text_features(args)
    train_img_features, val_img_features = load_image_features(args)
    hyperparams = get_hyperparameters(args, train_img_features, text_features)
    hyperparams_str = get_hyperparams_str(hyperparams)
    classifier_dir = get_classifier_dir(args, hyperparams_str)
    classifier_path = os.path.join(classifier_dir, "classifier.pth")
    model = SFUniDACLIP(text_features, img_encoder_path, classifier_path, args)
    return model


def create_optimizer(model, args):
    param_group = []
    for k, v in model.img_encoder.named_parameters():
        v.requires_grad = False
    for k, v in model.classifier.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    optimizer = torch.optim.AdamW(param_group)
    duplicate_init_lr(optimizer)
    return optimizer