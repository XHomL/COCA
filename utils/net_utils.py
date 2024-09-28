import random

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from tqdm import tqdm

from engine.datasets.utils import get_text_features_path

# Global variables
best_results = dict()

# Global constant
COEFF_LIST = [0.33333, 0.5, 1, 2, 3]


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def get_entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

@torch.no_grad()
def duplicate_init_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['init_lr'] = param_group['lr']


@torch.no_grad()
def update_lr(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['init_lr'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def get_performance_results(class_list,
                            pred_probs,
                            gt_labels,
                            open_flag=True,
                            open_thresh=0.5,
                            pred_unc_all=None):
    results = dict()
    img_num_per_cls = np.zeros((len(class_list)))
    correct_num_per_cls = np.zeros_like(img_num_per_cls)
    pred_labels = torch.max(pred_probs, dim=1)[1]  # [N]

    if open_flag:
        cls_num = pred_probs.shape[1]

        if pred_unc_all is None:
            # If there is not pred_unc_all tensor,
            # We normalize the Shannon entropy to [0, 1] to denote the uncertainty.
            pred_unc_all = get_entropy(pred_probs) / np.log(cls_num)  # [N]

        unc_idx = torch.where(pred_unc_all > open_thresh)[0]
        pred_labels[unc_idx] = cls_num  # set these pred results to unknown

    for i, label in enumerate(class_list):
        label_idx = torch.where(gt_labels == label)[0]
        correct_idx = torch.where(pred_labels[label_idx] == label)[0]
        img_num_per_cls[i] = float(len(label_idx))
        correct_num_per_cls[i] = float(len(correct_idx))

    results['per_cls_acc'] = correct_num_per_cls / (img_num_per_cls + 1e-5)

    if open_flag:
        results['common_cls_acc'] = results['per_cls_acc'][:-1].mean()
        results['unknown_cls_acc'] = results['per_cls_acc'][-1]
        results['hos'] = compute_h_score(results['common_cls_acc'], results['unknown_cls_acc'])
    else:
        results['common_cls_acc'] = correct_num_per_cls.sum() / (img_num_per_cls.sum() + 1e-5)
        results['unknown_cls_acc'] = 0.0
        results['hos'] = 0.0

    return results


def compute_h_score(known_cls_acc, unknown_cls_acc):
    h_score = 2 * known_cls_acc * unknown_cls_acc / (known_cls_acc + unknown_cls_acc + 1e-5)
    return h_score


@torch.no_grad()
def create_pseudo_labels_via_textual_prototypes(img_loader,
                                                model,
                                                args):
    args.logger.info("Generating pseudo labels via textual prototypes")
    model.eval()
    img_features = extract_image_features(img_loader, model)
    textual_protos = load_textual_prototypes(args)
    best_k = determine_best_k(args, img_features.cpu())
    psd_labels = create_pseudo_labels(args, best_k, img_features, textual_protos)
    evaluate_pseudo_labels(img_loader, psd_labels, args)
    return psd_labels


def evaluate_pseudo_labels(img_loader,
                           psd_labels,
                           args):
    gt_labels = load_labels(img_loader)
    img_pred_labels = psd_labels.argmax(dim=1).cpu()
    unk_indices = (get_entropy(psd_labels) > 0.05)
    img_pred_labels[unk_indices] = args.num_classes

    cls_list = args.target_class_list
    img_num_per_cls = np.zeros(len(cls_list))
    pred_num_per_cls = np.zeros_like(img_num_per_cls)
    correct_pred_per_cls = np.zeros_like(img_num_per_cls)
    for i, label in enumerate(cls_list):
        label_idx = torch.where(gt_labels == label)[0]
        correct_pred_idx = torch.where(img_pred_labels[label_idx] == label)[0]
        pred_num_per_cls[i] = float(len(torch.where(img_pred_labels == label)[0]))
        img_num_per_cls[i] = float(len(label_idx))
        correct_pred_per_cls[i] = float(len(correct_pred_idx))
    per_class_acc = correct_pred_per_cls / (img_num_per_cls + 1e-5)
    args.logger.info("PSD AVG ACC:\t" + "{:.4f}".format(np.mean(per_class_acc)))
    args.logger.info("PSD PER ACC:\t" + "\t".join(["{:.4f}".format(item) for item in per_class_acc]))
    args.logger.info("PER CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in img_num_per_cls]))
    args.logger.info("PRE CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in pred_num_per_cls]))
    args.logger.info("PRE ACC NUM:\t" + "\t".join(["{:.0f}".format(item) for item in correct_pred_per_cls]))


def create_pseudo_labels(args,
                         best_k,
                         img_features,
                         textual_protos):
    faiss_kmeans = faiss.Kmeans(args.feature_dim, best_k, niter=100, verbose=False, min_points_per_centroid=1,
                                gpu=False)
    proto_indices = torch.zeros((img_features.shape[0], args.num_classes))  # [N, C]
    logit_scale = torch.FloatTensor([args.logit]).exp()
    img_text_similarities = F.softmax(logit_scale * img_features @ textual_protos.cpu().t(), dim=-1)
    img_protos = get_image_prototypes(faiss_kmeans, img_features)
    for cls in range(args.num_classes):
        img_protos_cls_text_similarity = img_protos @ textual_protos[cls].t()
        cls_pos_img_proto_idx = img_protos_cls_text_similarity.argmax(0)
        cls_neg_img_protos = torch.cat((img_protos[:cls_pos_img_proto_idx], img_protos[cls_pos_img_proto_idx + 1:]),
                                       dim=0)
        img_cls_text_similarity = img_text_similarities.t()[cls, :].unsqueeze(1)
        img_cls_neg_protos_similarity = torch.einsum("nd, kd -> nk", img_features, cls_neg_img_protos.cpu())  # [N, K-1]

        img_cls_protos_similarity = torch.cat([img_cls_text_similarity, img_cls_neg_protos_similarity], dim=1)  # [N, K]
        torch.set_printoptions(profile="full")
        _, proto_idx = torch.max(img_cls_protos_similarity, dim=-1)
        proto_indices[:, cls] = proto_idx.cpu()
    prior_labels = F.one_hot(img_text_similarities.argmax(dim=-1), num_classes=args.num_classes)
    psd_labels = (proto_indices == 0).float() * prior_labels
    unk_img_indices = (torch.sum(psd_labels, dim=-1) == 0)
    unk_class_label = 1. / args.num_classes
    psd_labels[unk_img_indices, :] += unk_class_label
    psd_labels = psd_labels.cuda()
    return psd_labels


def get_image_prototypes(faiss_kmeans,
                         img_features):
    faiss_kmeans.train(img_features.numpy())
    img_protos = torch.from_numpy(faiss_kmeans.centroids).cuda()
    img_protos = img_protos / torch.norm(img_protos, p=2, dim=-1, keepdim=True)  # [K, D]
    return img_protos


def determine_best_k(args, img_features):
    best_score = 0 if args.k_deter in ['silhouette', 'ch'] else np.inf
    if args.fixed_k == 0:
        img_features_pca = TSNE(n_components=2, init="pca", random_state=0).fit_transform(img_features)
        global COEFF_LIST
        for coeff in COEFF_LIST:
            k_candidate = max(int(args.num_classes * coeff), 2)
            kmeans = KMeans(n_clusters=k_candidate, random_state=0).fit(img_features_pca)
            cluster_labels = kmeans.labels_
            if args.k_deter == 'silhouette':
                score = silhouette_score(img_features_pca, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_coeff = coeff
            elif args.k_deter == 'ch':
                score = calinski_harabasz_score(img_features_pca, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_coeff = coeff
            elif args.k_deter == 'db':
                score = davies_bouldin_score(img_features_pca, cluster_labels)
                if score < best_score:
                    best_score = score
                    best_coeff = coeff
        best_k = int(args.num_classes * best_coeff)
    else:
        best_k = args.fixed_k
    args.logger.info("Best K value:\t" + "{:.4f}".format(best_k))
    return best_k


def load_textual_prototypes(args):
    text_features_path = get_text_features_path(args)
    text_features = torch.load(text_features_path)
    _, text_label_idx = torch.sort(text_features['labels'])
    textual_protos = text_features['features'][text_label_idx].cuda()
    textual_protos = textual_protos / torch.norm(textual_protos, p=2, dim=-1, keepdim=True)
    return textual_protos


def extract_image_features(img_test_loader, model):
    img_features = []
    for _, img_test, _, _ in tqdm(img_test_loader, ncols=60):
        img_test = img_test.cuda()
        img_feature, img_logit = model(img_test)
        img_features.append(img_feature.cpu())
    img_features = torch.cat(img_features, dim=0)  # [N, D]
    img_features = img_features / torch.norm(img_features, p=2, dim=1, keepdim=True)
    return img_features


@torch.no_grad()
def test(args, model, img_loader_test, src_flg=False):
    model.eval()

    class_list = get_class_list(args, src_flg)
    open_flg = get_open_flag(args, src_flg)
    pred_probs = get_prediction_probabilities(img_loader_test, model)
    gt_labels = load_labels(img_loader_test)

    results = get_performance_results(
        class_list,
        pred_probs,
        gt_labels,
        open_flg,
        open_thresh=args.tau)
    print_current_results(results, args)
    print_best_results(results, args)


def print_current_results(results,
                          args):
    if args.task == 'PDA':
        results['os'] = 0.0
    else:
        results['os'] = results['per_cls_acc'].mean()
    args.logger.info(
        f'Current: HOS/H-score:{results["hos"]:.4f}, '
        f'Common class accuracy:{results["common_cls_acc"]:.4f}, '
        f'Unknown class accuracy:{results["unknown_cls_acc"]:.4f}, '
        f'OS value:{results["os"]:.4f}')


def print_best_results(results, args):
    global best_results
    if args.task == 'PDA' or args.task == 'CLDA':
        if results['common_cls_acc'] >= best_results.get('common_cls_acc', 0):
            best_results['hos'] = results['hos']
            best_results['common_cls_acc'] = results['common_cls_acc']
            best_results['unknown_cls_acc'] = results['unknown_cls_acc']
            best_results['os'] = results['os']
            # save_head(args, model)
    else:
        if results["hos"] >= best_results.get('hos', 0):
            best_results['hos'] = results['hos']
            best_results['common_cls_acc'] = results['common_cls_acc']
            best_results['unknown_cls_acc'] = results['unknown_cls_acc']
            best_results['os'] = results['os']
            # save_head(args, model)
    args.logger.info(
        f'Best   : HOS/H-score:{best_results["hos"]:.4f}, '
        f'Common class accuracy:{best_results["common_cls_acc"]:.4f}, '
        f'Unknown class accuracy:{best_results["unknown_cls_acc"]:.4f}, '
        f'OS value:{best_results["os"]:.4f}')


def get_open_flag(args, src_flg):
    if src_flg:
        open_flg = False
    else:
        open_flg = args.num_target_private_classes > 0
    return open_flg


def get_class_list(args, src_flg):
    if src_flg:
        return args.source_class_list
    else:
        return args.target_class_list


def get_prediction_probabilities(img_loader_test, model):
    pred_labels = []
    for _, img_test, _, _ in tqdm(img_loader_test, ncols=60):
        img_test = img_test.cuda()
        _, img_logit = model(img_test)
        pred_labels.append(img_logit.softmax(dim=-1).cpu())
    pred_labels = torch.cat(pred_labels, dim=0)  # [N, C]
    return pred_labels


def load_labels(img_loader_test):
    gt_labels = []
    for _, _, img_label, _ in tqdm(img_loader_test, ncols=60):
        gt_labels.append(img_label)
    gt_labels = torch.cat(gt_labels, dim=0)  # [N]
    return gt_labels


def evaluate_source_model(model, img_loader_test, args):
    notation = "\n=======================================================\n"
    notation += f"EVALUATING SOURCE MODEL ON THE TARGET:{args.target,} BASED ON SOURCE:{args.source}\n"
    notation += "======================================================="
    args.logger.info(notation)
    test(args, model, img_loader_test, src_flg=False)
