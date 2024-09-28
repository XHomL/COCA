import argparse


###########################
# Source Model Training Phase
###########################
def build_dir_config(parser):
    parser.add_argument("--data-dir",
                        type=str,
                        default="./data",
                        help="where the datasets are saved")
    parser.add_argument("--img-list-dir",
                        type=str,
                        default="./image_lists",
                        help="where the image lists are saved")
    parser.add_argument("--few-shot-items-dir",
                        type=str,
                        default="./few_shot_items",
                        help="where the few-shot image lists are saved")
    parser.add_argument("--feature-dir",
                        type=str,
                        default="./features",
                        help="where to save pre-extracted features")
    parser.add_argument("--record-dir",
                        type=str,
                        default="./source_models",
                        help="where to save source models")
    parser.add_argument("--img-list-file",
                        type=str,
                        default='image_unida_list.txt')


def build_dataset_config(parser):
    parser.add_argument("--dataset",
                        type=str,
                        help="Name of the dataset",
                        choices=["office31", "officehome", "visda", "domainnet"],
                        required=True)
    parser.add_argument("--source",
                        type=str,
                        help="name of the source domain",
                        default=None)
    parser.add_argument("--num-classes",
                        type=int,
                        default=None)
    parser.add_argument("--train-shot",
                        type=int,
                        default=16,
                        help="number of train shot")
    parser.add_argument("--val-shot",
                        type=int,
                        default=4,
                        help="number of val shot")
    parser.add_argument("--seed",
                        type=int,
                        default=1,
                        help="seed number")
    parser.add_argument("--preload-flag",
                        action="store_false",
                        help="whether to preload images to memory")


def build_model_config(parser):
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cuda", "cpu"],
                        help="device")
    parser.add_argument("--clip-encoder",
                        type=str,
                        default="ViT-B/16",
                        choices=["ViT-B/16", "RN50x16"],
                        help="specify the clip encoder to use")
    parser.add_argument("--img-layer-idx",
                        type=int,
                        default=0,
                        choices=[0],
                        help="specify how many image encoder layers to finetune. 0 means none.")
    parser.add_argument("--text-layer-idx",
                        type=int,
                        default=0,
                        choices=[0],
                        help="specify how many text encoder layers to finetune. 0 means none.")
    parser.add_argument("--classifier-head",
                        type=str,
                        default="linear",
                        choices=["linear",  # linear classifier
                                 "adapter",
                                 # 2-layer MLP with 0.2 residual ratio following CLIP-adapter + linear classifier
                                 ],
                        help="classifier head architecture")
    parser.add_argument("--classifier-init",
                        type=str,
                        default="zeroshot",
                        choices=["zeroshot",  # zero-shot/one-shot-text-based initialization
                                 "random",  # random initialization
                                 ],
                        help="classifier head initialization")
    parser.add_argument("--logit",
                        type=float,
                        default=4.60517,  # CLIP's default logit scaling
                        choices=[4.60517],
                        help="logit scale (exp(logit) is the inverse softmax temperature)")


def build_train_config(parser):
    parser.add_argument("--text-augmentation",
                        type=str,
                        default='vanilla',
                        choices=['vanilla',  # a photo of a {cls}.
                                 ],
                        help="specify the text augmentation to use.")
    parser.add_argument("--img-augmentation",
                        type=str,
                        default='flip',
                        choices=['none',  # only a single center crop
                                 'flip',  # add random flip view
                                 'randomcrop',  # add random crop view
                                 ],
                        help="specify the image augmentation to use.")
    parser.add_argument("--img-views",
                        type=int,
                        default=2,
                        help="if img-augmentation is not None, then specify the number of extra views.")
    parser.add_argument("--test-batch-size",
                        type=int,
                        default=32,
                        help="batch size for test (feature extraction and evaluation)")
    parser.add_argument("--num-workers",
                        type=int,
                        default=8,
                        help="number of workers for dataloader")
    parser.add_argument("--modality",
                        type=str,
                        default="cross_modal",
                        choices=["cross_modal",  # half batch image, half batch text
                                 "uni_modal",  # whole batch image
                                 ],
                        help="whether or not to perform cross-modal training, i.e. half batch is image, half batch is text)")
    parser.add_argument("--hyperparams",
                        type=str,
                        default="linear",
                        choices=["linear",  # linear hyper
                                 "adapter",  # adapter hyper
                                 ],
                        help="hyperparams sweep")
    parser.add_argument("--feature-dim",
                        type=int,
                        default=512,
                        help="feature dimension")

parser = argparse.ArgumentParser()
build_dir_config(parser)
build_dataset_config(parser)
build_model_config(parser)
build_train_config(parser)
