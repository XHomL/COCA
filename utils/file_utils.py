import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix

from engine.tools.utils import get_backbone_name


def cm_analysis(y_true, y_pred, filename, labels, classes, ymap=None, figsize=(17, 17)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    sns.set(font_scale=2.8)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / (cm_sum.astype(float) + 1e-8) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.0f%%\n%d/%d' % (p, c, s)
            # elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = '%.0f%%\n%d' % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True Label'
    cm.columns.name = 'Predicted Label'
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va='center')

    sns.heatmap(
        cm,
        annot=annot,
        fmt='',
        ax=ax,
        xticklabels=classes,
        cbar=True,
        cbar_kws={'format': PercentFormatter()},
        yticklabels=classes,
        cmap="Blues")
    plt.savefig(filename, bbox_inches='tight')


def get_log_dir(args):
    log_dir = os.path.join("logs",
                           "main",
                           args.modality,
                           args.classifier_head,
                           f"encoder_{get_backbone_name(args.clip_encoder)}",
                           args.dataset,
                           args.task,
                           f"shot_{args.train_shot}",
                           f"seed_{args.seed}",
                           f"num_classes_{args.num_classes}-tau_{args.tau}-lr_{args.lr}-wd_{args.weight_decay}-mr_{args.mask_ratio}",
                           f"src_{args.source}-tgt_{args.target}")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    return log_dir


def log_args(args):
    s = "\n==========================================\n"
    s += ("python" + " ".join(sys.argv) + "\n")
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    s += "==========================================\n"
    return s


def create_logger(args, log_name="train_log.txt"):
    log_dir = get_log_dir(args)
    print(f"log_dir: {log_dir}")
    # creating logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Init the logging file.
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode="w")

    file_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_format)

    # terminal logger handler
    terminal_handler = logging.StreamHandler()
    terminal_format = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    terminal_handler.setLevel(logging.INFO)
    terminal_handler.setFormatter(terminal_format)

    logger.addHandler(file_handler)
    logger.addHandler(terminal_handler)
    logger.debug(log_args(args))

    return logger
