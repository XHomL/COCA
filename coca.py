from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config.adaptation_config import build_adaptation_config
from dataset.utils import (create_text_features_loader,
                           create_img_loaders)
from model.teacher import EMATeacherCLIP
from model.utils import (load_source_model,
                         create_masking,
                         create_optimizer)
from utils.file_utils import create_logger
from utils.net_utils import (set_random_seed,
                             get_entropy,
                             update_lr,
                             create_pseudo_labels_via_textual_prototypes,
                             test,
                             evaluate_source_model)

# warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


def load_full_text_batch(text_features_loader, args):
    text_features_iter = iter(text_features_loader)
    text_feature, text_label, _ = next(text_features_iter)
    text_feature_batch = text_feature.cuda()
    text_label_batch = text_label.cuda()
    while text_feature_batch.shape[0] < args.batch_size:
        try:
            text_feature, text_label, _ = next(text_features_iter)
        except StopIteration:
            text_features_iter = iter(text_features_loader)
            text_feature, text_label, _ = next(text_features_iter)
        text_feature_batch = torch.cat((text_feature_batch, text_feature.cuda()), dim=0)
        text_label_batch = torch.cat((text_label_batch, text_label.cuda()), dim=0)
    text_feature_batch = text_feature_batch[:args.batch_size]
    text_label_batch = text_label_batch[:args.batch_size]

    return text_feature_batch, text_label_batch


def compute_image_loss(img, img_psd_label, model):
    _, logit = model(img)
    img_loss = torch.sum(-img_psd_label * torch.log(logit.softmax(dim=-1) + 1e-5), dim=-1).mean()
    return img_loss


def compute_text_loss(args, model, text_features_loader):
    feature, label = load_full_text_batch(text_features_loader, args)
    _, logit = model(feature, is_text=True)
    onehot_label = F.one_hot(label, num_classes=args.num_classes)
    text_loss = torch.sum(-onehot_label * torch.log(logit.softmax(dim=-1) + 1e-5), dim=-1).mean()
    return text_loss


def compute_masking_loss(img_idx,
                         img_psd_labels,
                         img,
                         iter_idx,
                         masking,
                         model,
                         teacher):
    teacher.update_weights(model, iter_idx)
    _, _, teacher_pred = teacher(img)
    masked_img = masking(img)
    _, masked_img_logit = model(masked_img)
    pos_mask = get_entropy(img_psd_labels[img_idx]) < 0.05
    pos_mask = pos_mask.long()
    ce_loss = torch.sum(-teacher_pred * torch.log(masked_img_logit.softmax(dim=-1) + 1e-5), dim=-1)
    mask_loss = torch.mean(ce_loss * pos_mask)
    return mask_loss


def get_average_losses(img_loss_list, mask_loss_list, text_loss_list, total_loss_list):
    avg_loss = dict()
    avg_loss["total_loss"] = np.mean(total_loss_list)
    avg_loss["img_loss"] = np.mean(img_loss_list)
    avg_loss["text_loss"] = np.mean(text_loss_list)
    avg_loss["mask_loss"] = np.mean(mask_loss_list)
    return avg_loss


def print_loss(loss_dict, epoch_idx, args):
    args.logger.info("Epoch: {}/{},          total_loss:{:.4f},\n\
                       image_loss:{:.4f}, text_loss:{:.4f}, mask_loss:{:.4f}".format(epoch_idx + 1, args.epochs,
                                                                                     loss_dict["total_loss"],
                                                                                     loss_dict["img_loss"],
                                                                                     loss_dict["text_loss"],
                                                                                     loss_dict["mask_loss"]))


def print_training_prompt(args):
    notation_str = "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.target,
                                                                                       args.source)
    notation_str += "======================================================="
    args.logger.info(notation_str)


def train(args,
          model,
          img_loader_train,
          text_feature_loader,
          optimizer,
          teacher: EMATeacherCLIP,
          masking,
          img_psd_labels,
          epoch_idx=0.0):
    model.train()
    total_loss_list, img_loss_list, text_loss_list, mask_loss_list = [], [], [], []
    iter_idx = epoch_idx * len(img_loader_train)
    iter_max = args.epochs * len(img_loader_train)
    for img_train, _, _, img_idx in tqdm(img_loader_train, ncols=60):
        iter_idx += 1
        img_train = img_train.cuda()
        img_idx = img_idx.cuda()
        img_psd_label = img_psd_labels[img_idx]  # [B, C]

        img_loss = compute_image_loss(img_train, img_psd_label, model)
        text_loss = compute_text_loss(args, model, text_feature_loader)
        mask_loss = compute_masking_loss(img_idx, img_psd_labels, img_train, iter_idx, masking, model, teacher)

        loss = img_loss + text_loss + mask_loss
        update_lr(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_list.append(loss.cpu().item())
        img_loss_list.append(img_loss.cpu().item())
        text_loss_list.append(text_loss.cpu().item())
        mask_loss_list.append(mask_loss.cpu().item())
    avg_losses = get_average_losses(img_loss_list, mask_loss_list, text_loss_list, total_loss_list)
    print_loss(avg_losses, epoch_idx, args)


def main(args):
    args.logger = create_logger(args, log_name="adaptation_log.txt")

    model = load_source_model(args).cuda()
    teacher = EMATeacherCLIP(model).cuda()
    masking = create_masking(args)
    optimizer = create_optimizer(model, args)

    text_features_loader = create_text_features_loader(args)
    train_imgs_loader, val_imgs_loader = create_img_loaders(args)

    evaluate_source_model(model, val_imgs_loader, args)

    print_training_prompt(args)

    model.eval()
    img_psd_labels = create_pseudo_labels_via_textual_prototypes(val_imgs_loader, model, args)
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        # Train on target domain
        train(args, model, train_imgs_loader, text_features_loader, optimizer, teacher, masking, img_psd_labels,
              epoch_idx)
        test(args, model, val_imgs_loader, src_flg=False)


if __name__ == "__main__":
    args = build_adaptation_config()
    args.model_name = Path(__file__).stem
    set_random_seed(args.seed)
    print(f'args.preload_flag: {args.preload_flag}')
    main(args)
