import argparse
import os

from engine.config import (build_dir_config,
                           build_dataset_config,
                           build_model_config,
                           build_train_config)
from engine.datasets.utils import validate_domain


def set_office31_config(args):
    args.source_dir = os.path.join("./image_lists/office31", args.source)
    args.target_dir = os.path.join("./image_lists/office31", args.target)
    args.num_common_classes = 10
    if args.task == "OPDA":
        args.num_source_private_classes = 10
        args.num_target_private_classes = 11
    else:
        raise NotImplementedError("Unknown target label type specified")


def set_officehome_config(args):
    args.source_dir = os.path.join("./image_lists/officehome", args.source)
    args.target_dir = os.path.join("./image_lists/officehome", args.target)
    if args.task == "PDA":
        args.num_common_classes, args.num_source_private_classes, args.num_target_private_classes = 25, 40, 0
    elif args.task == "OSDA":
        args.num_common_classes, args.num_source_private_classes, args.num_target_private_classes = 25, 0, 40
    elif args.task == "OPDA":
        args.num_common_classes, args.num_source_private_classes, args.num_target_private_classes = 10, 5, 50
    else:
        raise NotImplementedError("Unknown target label type specified")


def set_visda_config(args):
    args.source_dir = "./image_lists/visda/train/"
    args.target_dir = "./image_lists/visda/validation/"
    args.target_dir_list = [args.target_dir]
    args.source, args.target = 'train', 'validation'
    args.num_common_classes = 6
    if args.task == "PDA":
        args.num_source_private_classes, args.num_target_private_classes = 6, 0
    elif args.task == "OSDA":
        args.num_source_private_classes, args.num_target_private_classes = 0, 6
    elif args.task == "OPDA":
        args.num_source_private_classes, args.num_target_private_classes = 3, 3
    else:
        raise NotImplementedError(f"Unknown target label type specified: {args.task}")


def set_domainnet_config(args):
    args.source_dir = os.path.join("./image_lists/domainnet", args.source)
    args.target_dir = os.path.join("./image_lists/domainnet", args.target)
    args.num_common_classes = 150
    if args.task == "OPDA":
        args.num_source_private_classes, args.num_target_private_classes = 50, 145
    else:
        raise NotImplementedError("Unknown target label type specified")


def set_class_numbers(args):
    args.source_class_num = args.num_common_classes + args.num_source_private_classes
    args.target_class_num = args.num_common_classes + args.num_target_private_classes
    args.num_classes = args.source_class_num


def set_class_lists(args):
    args.source_class_list = list(range(args.source_class_num))
    args.target_class_list = list(range(args.num_common_classes))
    if args.num_target_private_classes > 0:
        args.target_class_list.append(args.source_class_num)


def set_learning_rate(args):
    if args.lr == 0.0:
        lr_map = {
            'office31': 0.005,
            'officehome': 0.001,
            'visda': 0.0001,
            'domainnet': 0.00001
        }
        args.lr = lr_map.get(args.dataset, 0.001)  # Default to 0.001 if dataset not found


def build_domain_related_config(args):
    assert args.source != args.target
    validate_domain(args.dataset, args.source)
    validate_domain(args.dataset, args.target)

    config_functions = {
        'office31': set_office31_config,
        'officehome': set_officehome_config,
        'visda': set_visda_config,
        'domainnet': set_domainnet_config
    }

    config_function = config_functions.get(args.dataset)
    if config_function:
        config_function(args)
    else:
        raise NotImplementedError(f"Unknown dataset specified: {args.dataset}")

    set_class_numbers(args)
    set_class_lists(args)
    set_learning_rate(args)

def build_general_adaptation_config(parser):
    parser.add_argument("--target",
                        type=str,
                        help="name of the target domain",
                        default=None)
    parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        help="number of epochs")
    parser.add_argument("--lr",
                        default=0.0,
                        type=float,
                        help="learning rate")
    parser.add_argument("--batch-size",
                        type=int,
                        default=64)
    parser.add_argument("--tau",
                        default=0.55,
                        type=float,
                        help="threshold for determining common and unknown class samples")
    parser.add_argument("--task",
                        default="OPDA",
                        choices=["OPDA", "OSDA", "PDA"],
                        type=str)
    parser.add_argument("--weight-decay",
                        type=float,
                        default=0.01)
    parser.add_argument("--mask-ratio",
                        type=float,
                        default=0.5)
    parser.add_argument("--fixed-k",
                        type=int,
                        default=0)
    parser.add_argument("--k-deter",
                        type=str,
                        choices=['silhouette', 'db', 'ch'],
                        default='silhouette')


def build_adaptation_config():
    parser = argparse.ArgumentParser()
    build_dir_config(parser)
    build_dataset_config(parser)
    build_model_config(parser)
    build_train_config(parser)
    build_general_adaptation_config(parser)
    args = parser.parse_args()
    build_domain_related_config(args)
    return args
