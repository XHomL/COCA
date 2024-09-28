import torch

from engine import clip
from engine.config import parser
from engine.datasets.utils import (save_text_features,
                                   save_image_features,
                                   load_few_shot_items)
from engine.model.utils import (save_text_encoder,
                                load_text_encoder,
                                load_image_encoder,
                                save_image_encoder,
                                extract_text_features,
                                extract_image_features)
from engine.tools.utils import set_random_seed, get_lab2cname


def main(args):
    set_random_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    clip_model, _ = clip.load(args.clip_encoder, jit=False)
    clip_model.float()
    clip_model.eval()
    save_text_encoder(clip_model, args)
    save_image_encoder(clip_model, args)

    lab2cname = get_lab2cname(args)
    text_encoder = load_text_encoder(args)
    text_features = extract_text_features(lab2cname, text_encoder, args)
    save_text_features(text_features, args)

    few_shot_items = load_few_shot_items(args)
    img_encoder = load_image_encoder(args)
    img_features = extract_image_features(few_shot_items, img_encoder, args)
    save_image_features(img_features, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
