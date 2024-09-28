import copy

import torch
from timm.models.layers import DropPath
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class EMATeacherCLIP(nn.Module):
    def __init__(self,
                 model,
                 pseudo_label_weight=None):
        super(EMATeacherCLIP, self).__init__()
        self.ema_model = copy.deepcopy(model)
        self.alpha = 0.999
        self.pseudo_label_weight = pseudo_label_weight
        if self.pseudo_label_weight == 'None':
            self.pseudo_label_weight = None

    def _init_ema_weights(self, model):
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(model.parameters())
        mcp = list(self.ema_model.parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, model, iter_idx):
        # alpha_teacher = min(1 - 1 / (iter_idx + 1), self.alpha)
        alpha_teacher = self.alpha
        for ema_param, param in zip(self.ema_model.classifier.parameters(),
                                    model.classifier.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def update_weights(self, model, iter_idx):
        # Init/update ema model
        if iter_idx == 0:
            self._init_ema_weights(model)
        if iter_idx > 0:
            self._update_ema(model, iter_idx)

    @torch.no_grad()
    def forward(
            self,
            target_img,
            text_feat=None,
            cross_modal=False):
        # Generate pseudo-label
        for m in self.ema_model.modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        if cross_modal:
            logit = self.ema_model([target_img, text_feat], cross_modal=True)
            teacher_pred = torch.softmax(logit.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(teacher_pred, dim=1)
        else:
            _, img_logit = self.ema_model(target_img)
            teacher_pred = torch.softmax(img_logit.detach(), dim=1)
            pseudo_prob, pseudo_label = torch.max(teacher_pred, dim=1)

        if self.pseudo_label_weight is None:
            pseudo_weight = torch.tensor(1.).cuda()
        elif self.pseudo_label_weight == 'prob':
            pseudo_weight = pseudo_prob
        else:
            raise NotImplementedError(self.pseudo_label_weight)

        return pseudo_label, pseudo_weight, teacher_pred
