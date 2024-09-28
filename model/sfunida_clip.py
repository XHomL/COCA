import argparse

import torch
import torch.nn as nn

from engine.model.utils import create_classifier


class SFUniDACLIP(nn.Module):
    """Class docstring for SFUniDACLIP."""

    def __init__(self,
                 text_features,
                 img_encoder_path,
                 classifier_path,
                 args: argparse):
        """Constructor for SFUniDACLIP."""

        super(SFUniDACLIP, self).__init__()
        self.num_classes = args.num_classes  # num_shared_classes + num_source_private_classes

        self.img_encoder = torch.load(img_encoder_path).feature_extractor.cuda()
        self.classifier = create_classifier(text_features, args)
        self.classifier.load_state_dict(torch.load(classifier_path))

        self.logit_scale = torch.FloatTensor([args.logit]).cuda()

        self.args = args

    def forward(self,
                inputs,
                is_text=False):
        if not is_text:
            # input_imgs [B, 3, H, W]
            img_feature = self.img_encoder(inputs)
            if self.args.img_layer_idx != 0:
                img_feature = self.partial_model(img_feature)
            logit = self.classifier(img_feature)

            return img_feature, logit
        else:
            text_feature = inputs
            logit = self.classifier(text_feature)

            return text_feature, logit
