import json
import os
import random
from collections import defaultdict
from typing import Dict, List

from engine.config import parser
from engine.datasets.utils import validate_domain
from engine.tools.utils import makedirs, set_random_seed, get_few_shot_setup_name


def group_by_class(img_list: List[str]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for item in img_list:
        label = item.split()[1]
        groups[label].append(item)
    return groups


def load_image_list(args):
    img_list = []
    list_file = open(os.path.join(args.img_list_dir,
                                  args.dataset,
                                  args.source,
                                  args.img_list_file), 'r').read().splitlines()
    for line in list_file:
        if int(line.split()[1]) < int(args.num_classes):
            path = f'{args.source}/{line}'
            img_list.append(path)

    return img_list


def get_random_indices(args,
                       items,
                       repeat=False):
    indices = list(range(len(items)))
    if len(items) >= args.train_shot + args.val_shot:
        train_indices = random.sample(indices, args.train_shot)
        val_indices = random.sample(set(indices) - set(train_indices), args.val_shot)
    elif len(items) <= args.train_shot:
        if repeat:
            train_indices = random.choices(indices, k=args.train_shot)
        else:
            train_indices = indices
        val_indices = train_indices
    else:
        train_indices = random.sample(indices, args.train_shot)
        val_indices = list(set(indices) - set(train_indices))
    return train_indices, val_indices


def create_few_shot_items(args, repeat: bool = False) -> Dict[str, Dict[str, List[Dict]]]:
    assert args.train_shot > 0, "Train shot must be positive"
    print(f"Creating a {args.train_shot}-shot train set")

    img_list = load_image_list(args)
    grouped_items = group_by_class(img_list)

    few_shot_items = {'train': {'img_list': []}, 'val': {'img_list': []}}

    for label, items in grouped_items.items():
        train_indices, val_indices = get_random_indices(args, items, repeat=repeat)
        for set_type, idx_list in [('train', train_indices), ('val', val_indices)]:
            few_shot_items[set_type]['img_list'].extend([
                {
                    "img_path": f'{args.data_dir}/{args.dataset}/{item.split()[0]}',
                    "label": int(item.split()[1]),
                    "classname": item.split()[0].split('/')[-2].lower().replace('_', ' ')
                }
                for item in [items[idx] for idx in idx_list]
            ])

    return few_shot_items


def save_few_shot_items(items: Dict, args):
    file_path = os.path.join(args.few_shot_items_dir,
                             args.dataset,
                             f'src_{args.source}-num_classes_{args.num_classes}',
                             f"{get_few_shot_setup_name(args.train_shot, args.seed)}.json")
    makedirs(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(items, f, indent=4, separators=(",", ": "))


def main(args):
    set_random_seed(args.seed)
    validate_domain(args.dataset, args.source)
    few_shot_items = create_few_shot_items(args)
    save_few_shot_items(few_shot_items, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
