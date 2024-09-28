import os
from copy import deepcopy

import torch
from tqdm import tqdm

from engine.config import parser
from engine.datasets.utils import load_text_features, \
    load_image_features, get_train_loaders, get_val_loader
from engine.model.utils import get_hyperparameters, create_optimizer, create_classifier, \
    create_lr_scheduler
from engine.tools.utils import makedirs, set_random_seed, get_classifier_dir, get_hyperparams_str

torch.set_num_threads(8)  # To maximize efficiency, please tune the number of threads for your machine

MAX_ITERS = 12800
EVAL_FREQ = 100  # Evaluate on val set per 100 iterations (for early stopping)


def get_experiment_count(hyperparams):
    count = len(hyperparams['weight_decay'])
    return count


def validate(logit_head,
             val_img_features_loader,
             device="cuda"):
    with torch.no_grad():
        logit_head.eval()
        val_acc = 0
        val_count = 0.
        for img_feature, image_label in val_img_features_loader:
            img_feature = img_feature.to(device)
            image_label = image_label.to(device)
            logit = logit_head(img_feature)
            pred = torch.argmax(logit, dim=1)
            val_acc += torch.sum(pred == image_label).item()
            val_count += image_label.size(0)
            img_feature.cpu()
        val_acc /= val_count
    return val_acc


def get_train_data(img_features_iter,
                   img_features_loader,
                   text_features_iter,
                   text_features_loader,
                   device):
    if img_features_iter is not None:
        try:
            img_feature, img_label = next(img_features_iter)
        except StopIteration:
            img_features_iter = iter(img_features_loader)
            img_feature, img_label = next(img_features_iter)
        img_feature = img_feature.to(device)
        img_label = img_label.to(device)
    else:
        img_feature = None
    if text_features_iter is not None:
        try:
            text_feature, text_label, eot_indices = next(text_features_iter)
        except StopIteration:
            text_features_iter = iter(text_features_loader)
            text_feature, text_label, eot_indices = next(text_features_iter)
        text_feature = text_feature.to(device)
        text_label = text_label.to(device)
    else:
        text_feature = None
    if img_feature is not None and text_feature is not None:
        feature = torch.cat([img_feature, text_feature], dim=0)
        label = torch.cat([img_label, text_label], dim=0)
    elif img_feature is not None:
        feature = img_feature
        label = img_label
    elif text_feature is not None:
        feature = text_feature
        label = text_label
    else:
        raise ValueError("Both image_features and text_features are None")
    return feature, label


def save_classifier(records,
                    args,
                    hyperparams_str):
    classifier_dir = get_classifier_dir(args, hyperparams_str)
    makedirs(classifier_dir)
    classifier_path = os.path.join(classifier_dir, "classifier.pth")
    if os.path.exists(classifier_path):
        print(f"Already exists: {hyperparams_str}")
    else:
        print(f"Saving classifier: {hyperparams_str}")
        torch.save(records['classifier'], classifier_path)


def create_train_iters(text_feature_loader, train_img_features_loader):
    if train_img_features_loader is None and text_feature_loader is None:
        raise ValueError("Both train_img_features_loader and text_features_loader are None")
    if train_img_features_loader is not None:
        img_features_iter = iter(train_img_features_loader)
    else:
        img_features_iter = None
    if text_feature_loader is not None:
        text_features_iter = iter(text_feature_loader)
    else:
        text_features_iter = None
    return img_features_iter, text_features_iter


def train(train_img_features_loader,
          text_feature_loader,
          img_feature_loader_val,
          classifier,
          optimizer,
          scheduler,
          eval_freq=EVAL_FREQ,
          device="cuda"):
    img_features_iter, text_features_iter = create_train_iters(text_feature_loader, train_img_features_loader)
    criterion = torch.nn.CrossEntropyLoss()

    records = {
        "iter": None,
        "val_acc": None,
        "img_encoder": None,
        "text_encoder": None,
        "classifier": None
    }

    for i in tqdm(range(MAX_ITERS), ncols=80):
        classifier.train()
        feature, label = get_train_data(img_features_iter,
                                        train_img_features_loader,
                                        text_features_iter,
                                        text_feature_loader,
                                        device)

        logit = classifier(feature)
        loss = criterion(logit, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % eval_freq == 0:
            val_acc = validate(classifier, img_feature_loader_val, device=device)
            print(f"Iteration: {i}    Validation Accuracy: {val_acc:.4f}")
            if records["val_acc"] is None or val_acc > records["val_acc"]:
                records["iter"] = i
                records["val_acc"] = val_acc
                records["classifier"] = deepcopy(classifier.state_dict())

    print(f"Best val acc: {records['val_acc']:.4f} at iter {records['iter']}")
    return records


def main(args):
    set_random_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    text_features = load_text_features(args)
    train_img_features, val_img_features = load_image_features(args)
    hyperparams = get_hyperparameters(args, train_img_features, text_features)
    hyperparams_str = get_hyperparams_str(hyperparams)

    classifier = create_classifier(text_features, args)
    optimizer = create_optimizer(classifier, hyperparams)
    lr_scheduler = create_lr_scheduler(optimizer, hyperparams)

    train_img_features_loader, text_features_loader = get_train_loaders(train_img_features,
                                                                        text_features,
                                                                        args,
                                                                        hyperparams)

    val_img_features_loader = get_val_loader(val_img_features,
                                             args,
                                             hyperparams)
    print('Start training')
    records = train(train_img_features_loader,
                    text_features_loader,
                    val_img_features_loader,
                    classifier,
                    optimizer,
                    lr_scheduler,
                    eval_freq=EVAL_FREQ)

    save_classifier(records, args, hyperparams_str)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
