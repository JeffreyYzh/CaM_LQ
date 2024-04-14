"""
Two-stage HOI detector with enhanced visual context

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import torchvision.ops.boxes as box_ops
from ops import recover_boxes
from PIL import Image

from torch import nn, Tensor
from collections import OrderedDict
from typing import Optional, Tuple, List
from torchvision.ops import FeaturePyramidNetwork
import clip
from hicodet.static_hico import ACT_IDX_TO_ACT_NAME, HICO_INTERACTIONS, ACT_TO_ING, HOI_IDX_TO_ACT_IDX, \
    HOI_IDX_TO_OBJ_IDX, MAP_AO_TO_HOI, UA_HOI_IDX

from transformers import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    SwinTransformer,
)

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)

from detr.models import build_model as build_base_detr
from h_detr.models import build_model as build_advanced_detr
from detr.models.position_encoding import PositionEmbeddingSine
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list

class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size[:-1], repr_size[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:  # x:270,512;y:270,384
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, obj_to_verb, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size  # 384
        self.num_verbs = num_verbs  # 117
        self.human_idx = human_idx  # 0
        self.obj_to_verb = obj_to_verb

        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        n_h = torch.sum(is_human)
        if not torch.all(labels[:n_h]==self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return n_h

    def compute_box_pe(self, boxes, embeds, image_size):
        bx_norm = boxes / image_size[[1, 0, 1, 0]]  # 获得归一化的box及center坐标, wh
        bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2    # n,2
        b_wh = bx_norm[:, 2:] - bx_norm[:, :2]  # n,2

        c_pe = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)  # n, 256    bx_x[:, None]:n,1,2
        wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1) # n, 256

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)   # 19,512

        # Modulate the positional embeddings with box widths and heights by
        # applying different temperatures to x and y
        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()    # n_query, 2,获得论文中的wref\href
        # Note that the positional embeddings are stacked as [pe(y), pe(x)]对中心点的位置编码采用不同的温度参数,温度参数根据不同的高宽来设置
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        return box_pe, c_pe

    def forward(self, region_props, image_sizes, device=None):  # todo:加上对union box的编码
        if device is None:
            device = region_props[0]["hidden_states"].device

        ho_queries = []
        paired_indices = []
        prior_scores = []
        object_types = []
        positional_embeds = []
        for i, rp in enumerate(region_props):
            boxes, scores, labels, embeds = rp.values()
            nh = self.check_human_instances(labels) # human的数量15
            n = len(boxes)  # 19
            # Enumerate instance pairs
            x, y = torch.meshgrid(  # (19,19)
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < nh)).unbind(1) # x只保留box label为human的,0~14
            # Skip image when there are no valid human-object pairs
            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({})
                continue
            x = x.flatten(); y = y.flatten()
            # Compute spatial features
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x],], [boxes[y],], [image_sizes[i],]
            )   # 计算空间特征361(19*19),36
            pairwise_spatial = self.spatial_head(pairwise_spatial)  # 更改维度,计算位置特征361(19*19),384
            pairwise_spatial_reshaped = pairwise_spatial.reshape(n, n, -1)  # 19,19,384

            box_pe, c_pe = self.compute_box_pe(boxes, embeds, image_sizes[i])   # 计算空间特征(19,512),(19,256)，前者为中心点和宽高结合的空间特征，后者为中心点正弦位置编码
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))  # 结合空间特征和隐藏层特征构建embedding
            embeds = embeds.squeeze(1)
            # Compute human-object queries
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),
                pairwise_spatial_reshaped[x_keep, y_keep]   # query特征和空间特征被一起送进去
            )   # 得到编码后的query
            # Append matched human-object pairs
            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels, self.num_verbs, self.training,
                self.obj_to_verb
            ))  # 270,2,117
            object_types.append(labels[y_keep])
            positional_embeds.append({
                "centre": torch.cat([c_pe[x_keep], c_pe[y_keep]], dim=-1).unsqueeze(1), # 5,1,1024
                "box": torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds

class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims
    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)

class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )
    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)
        )
        mask = x[self.return_layer].mask
        x = self.fpn(pyramid)[f"{self.return_layer}"]
        x = self.layers(x)
        return x, mask

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

class PViC(nn.Module):
    """Two-stage HOI detector with enhanced visual context"""

    def __init__(self,
        detector: Tuple[nn.Module, str], postprocessor: nn.Module,
        feature_head: nn.Module, ho_matcher: nn.Module,
        triplet_decoder: nn.Module, num_verbs: int,
        clip_model:nn.Module, preprocess,
        text_features,
        repr_size: int = 384, human_idx: int = 0,
        # Focal loss hyper-parameters
        alpha: float = 0.5, gamma: float = .1,
        # Sampling hyper-parameters
        box_score_thresh: float = .05,
        min_instances: int = 3,
        max_instances: int = 15,
        raw_lambda: float = 2.8,
    ) -> None:
        super().__init__()

        self.detector = detector[0]
        self.od_forward = {     # object detector forward
            "base": self.base_forward,
            "advanced": self.advanced_forward,
        }[detector[1]]
        self.postprocessor = postprocessor

        self.ho_matcher = ho_matcher
        self.feature_head = feature_head
        self.kv_pe = PositionEmbeddingSine(128, 20, normalize=True)
        self.decoder = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.repr_size = repr_size
        self.human_idx = human_idx
        self.num_verbs = num_verbs
        self.alpha = alpha
        self.gamma = gamma
        self.box_score_thresh = box_score_thresh
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.raw_lambda = raw_lambda
        # add clip
        self.clip = clip_model
        # self.topk_is = args.topk_is
        # self.gtclip = args.gtclip
        # self.neg_0 = args.neg_0
        self.preprocess = preprocess
        self.text_features = text_features

        self.nointer_mask = None

    def freeze_detector(self):
        for p in self.detector.parameters():
            p.requires_grad = False

    def compute_classification_loss_w_clip1(self, logits, prior, labels, weights=None, clipseen_weights=None, alpha=0.25):
        prior = torch.cat(prior, dim=0).prod(
            1)  # 438,117.prod返回第一维度所有元素的乘积,也就是human score和object score对应乘积;乘积不为0表示可能出现的verb
        x, y = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]  # 只拿所有可能出现的verb logits进行比较
        prior = prior[x, y]  # 18052;得到所有可能的verb对应的分数
        labels = labels[None, x, y].repeat(len(logits), 1)

        pred = torch.log(
            prior / (1 + torch.exp(-logits) - prior) + 1e-8
        ).sigmoid()

        pos_inds = labels.eq(1).float()
        neg_inds = labels.eq(0).float()
        soft_inds = torch.logical_and(labels.gt(0), labels.lt(1)).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds  # focal loss:gamma=2
        if weights is not None:
            pos_loss = pos_loss * weights[:-1]

        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        soft_loss = F.binary_cross_entropy(pred, labels, reduction='none') * soft_inds
        if clipseen_weights is not None:
            soft_loss = soft_loss * clipseen_weights[:-1]

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        soft_loss = soft_loss.mean()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss, soft_loss

    def compute_classification_loss_w_clip2(self, logits, prior, labels, clipseen_weights=None):
        prior = torch.cat(prior, dim=0).prod(
                1)  # 438,117.prod返回第一维度所有元素的乘积,也就是human score和object score对应乘积;乘积不为0表示可能出现的verb
        x, y = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]  # 只拿所有可能出现的verb logits进行比较
        prior = prior[x, y]  # 18052;得到所有可能的verb对应的分数
        labels = labels[None, x, y].repeat(len(logits), 1)

        gt_inds = torch.logical_or(labels.eq(1), labels.eq(0)).float()
        pos_inds = labels.eq(1).float()
        soft_inds = torch.logical_and(labels.gt(0), labels.lt(1)).float()

        n_p = (labels * pos_inds).sum()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        gt_loss = binary_focal_loss_with_logits(torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='none') * gt_inds
        gt_loss = gt_loss.sum()

        soft_loss = F.binary_cross_entropy(torch.log(
            prior / (1 + torch.exp(-logits) - prior) + 1e-8
        ).sigmoid(), labels, reduction='none') * soft_inds
        if clipseen_weights is not None:
            soft_loss = soft_loss * clipseen_weights[:-1]

        soft_loss = soft_loss.mean()

        return gt_loss / n_p, soft_loss

    def postprocessing(self,boxes, paired_inds, object_types,logits, prior, image_sizes):
        n = [len(p_inds) for p_inds in paired_inds]
        logits = logits.split(n)

        detections = []
        for bx, p_inds, objs, lg, pr, size in zip(
            boxes, paired_inds, object_types,
            logits, prior, image_sizes):
            pr = pr.prod(1)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y].sigmoid() * pr[x, y].pow(self.raw_lambda)
            detections.append(dict(boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size, x=x))
        return detections

    @staticmethod
    def base_forward(ctx, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = ctx.transformer(ctx.input_proj(src), mask, ctx.query_embed.weight, pos[-1])[0]

        outputs_class = ctx.class_embed(hs)
        outputs_coord = ctx.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out, hs, features

    @staticmethod
    def advanced_forward(ctx, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(ctx.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if ctx.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, ctx.num_feature_levels):
                if l == _len_srcs:
                    src = ctx.input_proj[l](features[-1].tensors)
                else:
                    src = ctx.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = ctx.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not ctx.two_stage or ctx.mixed_selection:
            query_embeds = ctx.query_embed.weight[0 : ctx.num_queries, :]

        self_attn_mask = (
            torch.zeros([ctx.num_queries, ctx.num_queries,]).bool().to(src.device)
        )
        self_attn_mask[ctx.num_queries_one2one :, 0 : ctx.num_queries_one2one,] = True
        self_attn_mask[0 : ctx.num_queries_one2one, ctx.num_queries_one2one :,] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = ctx.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = ctx.class_embed[lvl](hs[lvl])
            tmp = ctx.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes_one2one.append(outputs_class[:, 0 : ctx.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, ctx.num_queries_one2one :])
            outputs_coords_one2one.append(outputs_coord[:, 0 : ctx.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, ctx.num_queries_one2one :])
        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)
        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
        }

        if ctx.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord,
            }
        return out, hs, features

    def generate_union_box(self, boxes, paired_inds):
        boxes = [boxes[i][ids] for i, ids in enumerate(paired_inds)]
        # 获取左上角和右下角坐标
        box1_tl, box1_br = boxes[:, 0, :2], boxes[:, 0, 2:]
        box2_tl, box2_br = boxes[:, 1, :2], boxes[:, 1, 2:]

        # 计算联合框的左上角和右下角坐标
        union_tl = torch.min(box1_tl, box2_tl)
        union_br = torch.max(box1_br, box2_br)

        # 组合左上角和右下角坐标形成联合框
        union_boxes = torch.cat([union_tl, union_br], dim=1)

        return union_boxes

    def generate_pos_idx(self, boxes, paired_inds, targets, thresh=0.5):
        """
        增加clip作为预测标签
        """
        # num_classes = self.num_verbs
        # labels = []
        pos_match_idx = []
        for bx, p_inds, target in zip(boxes, paired_inds, targets):
            # is_match = torch.zeros(len(p_inds), num_classes, device=bx.device)

            bx_h, bx_o = bx[p_inds].unbind(1)
            gt_bx_h = recover_boxes(target["boxes_h"], target["size"])
            gt_bx_o = recover_boxes(target["boxes_o"], target["size"])

            x, y = torch.nonzero(torch.min(
                box_ops.box_iou(bx_h, gt_bx_h),
                box_ops.box_iou(bx_o, gt_bx_o)
            ) >= thresh).unbind(1)  # 只有box重合大于0.5才分配,否则作为负样本

            # is_match[x, target["labels"][y]] = 1
            # labels.append(is_match)
            # add 匹配上的正样本
            pos_match_idx.append((x, y))
        return pos_match_idx
        # return pos_match_idx, labels

    def compute_target_label(self, logits, targets, indices, num_pred_ind):
        """

        Args:
            logits: (2,1730,117)
            targets: list of dict
            indices: list(tuple(2))
            priors:list(tensor(n,2,117), tensor(,,),...)

        Returns:

        """
        src_logits = logits # (2,1730,117)
        # todo:改成线下crop,forward的时候直接计算
        ps_idxss = []
        objs = []
        unions = []
        for t, indice in zip(targets, indices):
            ps_idxs = []  # 该张图像匹配上的实例对应target的下标列表
            # or_ps_idxs = [i for i in range(len(t['st']))]  # st是干嘛用的？seen target可见类别！
            or_ps_idxs = [i for i in range(len(t['verb']))]

            for idx in or_ps_idxs:
                if idx in indice[1]:
                    ps_idxs.append(idx)

            ps_idxss.append(ps_idxs)
            obj_labels = t['object']  # 加上obj label和union box 坐标
            objs.extend(obj_labels[ps_idxs])
            unions.append(t['union_tensors'][ps_idxs])

        regions = torch.cat(unions) # woc竟然是用GT box来crop的

        target_classes = torch.zeros_like(src_logits[-1], dtype=torch.float32, device=logits.device)    # 多个tuple,每个tuple为一张图像预测实例的verb标签

        # clip forward
        regions = torch.as_tensor(regions).to(src_logits.device)  # 11,117
        if len(regions):
            image_features = self.clip.encode_image(regions)  # 11,768, todo:用更多的CLIP处理的特征
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 11,768
            logit_scale = self.clip.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features.t()  # 11,600
            # refer object softmax
            hoi2obj = torch.as_tensor(HOI_IDX_TO_OBJ_IDX).unsqueeze(0).repeat(logits.shape[0], 1)  # 11,600
            objs1 = torch.as_tensor(objs, device=hoi2obj.device).unsqueeze(1).repeat(1, logits.shape[1])  # 11,600
            hoimask = (hoi2obj == objs1).to(logits.device)  # (n, 600) 找到obj可能的交互类别
            # if self.nointer_mask is not None:  # 将无交互的类别置为False
            #     nointer = self.nointer_mask.unsqueeze(0).repeat(logits.shape[0], 1).to(logits.device)
            #     hoimask[nointer == True] = False
            logits = logits.masked_fill(hoimask == False, float('-inf'))    # 11,600
            logits = logits.softmax(dim=-1)  # 得到当前regions可能的交互类别(11,600)
            # map hoi to action
            objs2 = torch.as_tensor(objs).unsqueeze(-1).repeat(1, self.num_verbs)  # 11,117
            actions = torch.arange(self.num_verbs).unsqueeze(0).repeat(logits.shape[0], 1)  # 11,117
            ind = torch.as_tensor(MAP_AO_TO_HOI, device=logits.device)[
                actions, objs2]  # 常用于根据提供的索引选择张量的特定元素或子数组,将action和object的组合映射成hoi  11,117
            zeros = torch.zeros((logits.shape[0], 1), device=logits.device)  # 11,1
            logits = torch.cat([logits, zeros], dim=-1)  # 11, 601
            new_logits = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                    zip(logits, ind)])  # index_select从张量 a 中按照索引 i 进行选择,(11,117),选择出sigma\n中每一个实例的clip交互分数

            num_preds_per_image = [len(p) for p in ps_idxss]  # list(4)
            logitss = new_logits.split(num_preds_per_image, dim=0)  # tuple((7,117),(),(),())

            for t, logits, ps_idxs in zip(targets, logitss, ps_idxss):
                verb = torch.zeros((len(t['verb']), self.num_verbs), dtype=torch.float32, device=t['verb'].device)
                id = torch.arange(len(t['verb']))
                # assert max(t['verb']) <= 23, "out of bound"
                verb[id, t['verb']] = 1.
                t['verb'] = verb
                for ps_idx, logit in zip(ps_idxs, logits):
                    # if self.neg_0:
                    # logit = logit * t['seen_mask']  # 只能看见可见的
                    t['verb'][ps_idx] = torch.where(t['verb'][ps_idx] == 1., t['verb'][ps_idx],
                                                           logit)  # 选择gt标签,如果GT标签中有的就拿GT标签,反之拿logit标签
                    # t['verb'][ps_idx] = logit

            # idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t['verb'][J] for t, (_, J) in zip(targets, indices)]) # 9,117原始的标签

            add_num = [0]    # 后面赋值的时候需要添加的数
            st = 0
            for i in range(1, len(num_pred_ind)):
                st += num_pred_ind[i-1]
                add_num.append(st)

            idx = torch.cat([a_n + id for a_n, (id, _) in zip(torch.LongTensor(add_num), indices)]).to(target_classes_o).long() # 为了一次性地赋值
            if target_classes_o.shape[0]:
                target_classes[idx] = target_classes_o
        return target_classes

    def compute_verb_label(self, logits, targets, indices, num_pred_ind):
        """

        Args:
            logits: (2,1730,117)
            targets: list of dict
            indices: list(tuple(2))
            labels: list(tenosr(n,117))
            priors:list(tensor(n,2,117), tensor(,,),...)

        Returns:

        """
        src_logits = logits # (2,1730,117)
        # todo:改成线下crop,forward的时候直接计算
        regions = []
        objs = []
        ps_idxss = []
        assert len(targets) == len(indices), 'targets len != indices lens'
        for t, indice in zip(targets, indices):
            ps_idxs = []  # 该张图像匹配上的实例下标列表
            # or_ps_idxs = [i for i in range(len(t['st']))]  # st是干嘛用的？seen target可见类别！
            or_ps_idxs = [i for i in range(len(t['verb']))]

            for idx in or_ps_idxs:
                if idx in indice[1]:
                    ps_idxs.append(idx)
            img_or = t['org_img']    # img_origin需要加上
            assert img_or, "img_or is None"

            ps_idxss.append(ps_idxs)
            union_boxes = t['union_boxes']  # woc竟然是用GT box来crop的
            obj_labels = t['object']  # 加上obj label和union box 坐标

            for ps_idx in ps_idxs:
                ho_box = union_boxes[ps_idx]
                obj = obj_labels[ps_idx]
                x, y, x2, y2 = ho_box.cpu().numpy()
                region = img_or.crop((x, y, x2, y2))
                # pad
                region = expand2square(region, (0, 0, 0))
                region = self.preprocess(region).unsqueeze(0)
                regions.append(region)
                objs.append(obj)

        if regions != []:
            # clip forward
            regions = torch.cat(regions)
            regions = torch.as_tensor(regions).to(src_logits.device)  # 11,3,384,384
            image_features = self.clip.encode_image(regions)  # 11,768
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # 11,768
            logit_scale = self.clip.logit_scale.exp()
            logits = logit_scale * image_features @ self.text_features.t()  # 11,600
            # refer object softmax
            hoi2obj = torch.as_tensor(HOI_IDX_TO_OBJ_IDX).unsqueeze(0).repeat(logits.shape[0], 1)  # 11,600
            objs1 = torch.as_tensor(objs).unsqueeze(1).repeat(1, logits.shape[1])  # 11,600
            hoimask = (hoi2obj == objs1).to(logits.device)  # (n, 600) 找到obj可能的交互类别
            # if self.nointer_mask is not None:  # 将无交互的类别置为False
            #     nointer = self.nointer_mask.unsqueeze(0).repeat(logits.shape[0], 1).to(logits.device)
            #     hoimask[nointer == True] = False
            logits = logits.masked_fill(hoimask == False, float('-inf'))
            logits = logits.softmax(dim=-1)  # 得到当前regions可能的交互类别(11,600)
            # map hoi to action
            objs2 = torch.as_tensor(objs).unsqueeze(-1).repeat(1, self.num_verbs)  # 11,117
            actions = torch.arange(self.num_verbs).unsqueeze(0).repeat(logits.shape[0], 1)  # 11,117
            ind = torch.as_tensor(MAP_AO_TO_HOI, device=logits.device)[
                actions, objs2]  # 常用于根据提供的索引选择张量的特定元素或子数组,将action和object的组合映射成hoi
            zeros = torch.zeros((logits.shape[0], 1), device=logits.device)  # 11,1
            logits = torch.cat([logits, zeros], dim=-1)  # 11，601
            new_logits = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in
                                    zip(logits, ind)])  # index_select从张量 a 中按照索引 i 进行选择,(11,117)

            num_preds_per_image = [len(p) for p in ps_idxss]  # list(4)
            logitss = new_logits.split(num_preds_per_image, dim=0)  # tuple((7,117),(),(),())

            for t, logits, ps_idxs in zip(targets, logitss, ps_idxss):
                verb = torch.zeros((len(t['verb']), self.num_verbs), dtype=torch.float32, device=t['verb'].device)
                id = torch.arange(len(t['verb']))
                verb[id, t['verb']] = 1.
                t['verb'] = verb
                for ps_idx, logit in zip(ps_idxs, logits):
                    # if self.neg_0:
                    # logit = logit * t['seen_mask']  # 只能看见可见的
                    t['verb'][ps_idx] = torch.where(t['verb'][ps_idx] == 1., t['verb'][ps_idx],
                                                           logit)  # 选择gt标签,如果GT标签中有的就拿GT标签,反之拿logit标签

        # idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb'][J] for t, (_, J) in zip(targets, indices)]) # 9,117原始的标签
        target_classes = torch.zeros_like(src_logits[-1], dtype=torch.float32, device=logits.device)    # 多个tuple,每个tuple为一张图像预测实例的verb标签

        add_num = [0]    # 后面赋值的时候需要添加的数
        st = 0
        for i in range(1, len(num_pred_ind)):
            st += num_pred_ind[i-1]
            add_num.append(st)

        idx = torch.cat([a_n + id for a_n, (id, _) in zip(torch.LongTensor(add_num), indices)]).to(target_classes_o).long()
        if target_classes_o.shape[0]:
            target_classes[idx] = target_classes_o
        return target_classes

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def compute_classification_loss(self, logits, prior, labels):
        prior = torch.cat(prior, dim=0).prod(1)
        x, y = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]
        prior = prior[x, y]
        labels = labels[None, x, y].repeat(len(logits), 1)

        n_p = labels.sum()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )

        return loss / n_p

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (M, 2) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `size`: torch.Tensor
                (2,) Image height and width
            `x`: torch.Tensor
                (M,) Index tensor corresponding to the duplications of human-objet pairs. Each
                pair was duplicated once for each valid action.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        image_sizes = torch.as_tensor([im.size()[-2:] for im in images], device=images[0].device)

        with torch.no_grad():
            results, hs, features = self.od_forward(self.detector, images)  # 固定产生100个box(4,100,4)及其class(4,100,81)和embedding,hs有6层特征(6,4,100,256)
            results = self.postprocessor(results, image_sizes)  # score(100,),labels(100,),boxes(100,4)

        region_props = prepare_region_proposals(    # hs是hidden states,取最后一层的特征
            results, hs[-1], image_sizes,
            box_score_thresh=self.box_score_thresh,     # 过滤掉低置信度的box
            human_idx=self.human_idx,
            min_instances=self.min_instances,
            max_instances=self.max_instances
        )   # 得到过滤掉之后的region字典:boxes(N,4),scores(N),labels(N,),hidden_states(N,256)
        boxes = [r['boxes'] for r in region_props]  # 所有保留的box,前面的是human后面的是obj
        # Produce human-object pairs.
        (
            ho_queries,     # list:(15,384),(120,384)
            paired_inds, prior_scores,  # list(15,2) (120,2); list(15,2,117) (120,2,117);
            object_types, positional_embeds # list(15,) (120,),    [center:(120,1,512),box:(120,1,1024)]
        ) = self.ho_matcher(region_props, image_sizes)

        # add union boxes
        # union_boxes = self.generate_union_box(boxes, paired_inds)

        # Compute keys/values for triplet decoder.
        memory, mask = self.feature_head(features)  # (2,25,38,256),(2,25,38)
        b, h, w, c = memory.shape
        memory = memory.reshape(b, h * w, c)
        kv_p_m = mask.reshape(-1, 1, h * w)
        k_pos = self.kv_pe(NestedTensor(memory, mask)).permute(0, 2, 3, 1).reshape(b, h * w, 1, c)  # (2,950,1,256)
        # Enhance visual context with triplet decoder.
        query_embeds = []
        for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
            query_embeds.append(self.decoder(
                ho_q.unsqueeze(1),              # (n, 1, q_dim)
                mem.unsqueeze(1),               # (hw, 1, kv_dim)
                kv_padding_mask=kv_p_m[i],      # (1, hw)
                q_pos=positional_embeds[i],     # centre: (n, 1, 2*kv_dim), box: (n, 1, 4*kv_dim)
                k_pos=k_pos[i]                  # (hw, 1, kv_dim)
            ).squeeze(dim=2))
        # Concatenate queries from all images in the same batch.
        query_embeds = torch.cat(query_embeds, dim=1)     # (ndec, \sigma{n}, q_dim)    list(2,435,384)将最后两层decoder输出,所有预测的embedding concat起来
        logits = self.binary_classifier(query_embeds)   # (ndec,m,117)

        if self.training:
            labels, indices = associate_with_ground_truth(boxes, paired_inds, targets, self.num_verbs)
            # indices, labels = self.generate_pos_idx(boxes, paired_inds, targets, self.num_verbs)    # indices是和label匹配的关系,labels是用target分配的标签
            num_pred_ins = [len(p_id) for p_id in paired_inds]  # 每张图片预测出instance数量,最后拿来match给label会使用
            labels = self.compute_target_label(logits, targets, indices, num_pred_ins)    # (n,117)用来加入clip对称信息
            # cls_loss = self.compute_classification_loss1(logits, prior_scores, target_classes)
            verb_loss, clip_loss = self.compute_classification_loss_w_clip2(logits, prior_scores, labels)
            # loss_dict = dict(verb_loss=verb_loss, clip_loss=clip_loss)
            loss_dict = dict(verb_loss=verb_loss, clip_loss=clip_loss)
            # verb_loss = self.compute_classification_loss(logits, prior_scores, labels)
            # loss_dict = dict(verb_loss=verb_loss)
            return loss_dict

        detections = self.postprocessing(
            boxes, paired_inds, object_types,
            logits[-1], prior_scores, image_sizes
        )
        return detections

def build_detector(args, obj_to_verb):
    if args.detector == "base":
        detr, _, postprocessors = build_base_detr(args)
    elif args.detector == "advanced":
        detr, _, postprocessors = build_advanced_detr(args)

    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])

    ho_matcher = HumanObjectMatcher(
        repr_size=args.repr_dim,
        num_verbs=args.num_verbs,
        obj_to_verb=obj_to_verb,
        dropout=args.dropout
    )
    decoder_layer = TransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    triplet_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.triplet_dec_layers
    )
    return_layer = {"C5": -1, "C4": -2, "C3": -3}[args.kv_src]
    if isinstance(detr.backbone.num_channels, list):
        num_channels = detr.backbone.num_channels[-1]
    else:
        num_channels = detr.backbone.num_channels
    feature_head = FeatureHead(
        args.hidden_dim, num_channels,
        return_layer, args.triplet_enc_layers
    )

    # add clip
    device = torch.device(args.device)
    # build_clip
    if args.eval:
        text_features = None
        clip_model = None
        preprocess = None
        clip_model_w_adapter = None
    else:
        clip_model, preprocess = clip.load(args.clip_backbone, device=device)
        
        # if adapter
        # clip_model_w_adapter = My_Model(clip_model)
        clip_model_w_adapter = My_Model(clip_model)
        clip_model_w_adapter.to(device)

        print(f"The backbone of CLIP is {args.clip_backbone}.")
        print("Turning off gradients in both the image and the text encoder")
        for name, param in clip_model_w_adapter.named_parameters():
            param.requires_grad_(False)

        if args.pretrain_clip:
            clip_model_w_adapter.load_state_dict(torch.load(args.pretrain_clip)['model'], strict=False)
            print(f"CLIP is initialized from {args.pretrain_clip}.")
        # if args.pretrain_clip:
        #     region_clip = torch.load(args.pretrain_clip)['model']
        #     for key in list(region_clip.keys()):
        #         if key.startswith('teacher_backbone'):
        #             region_clip[('visual' + key[len('teacher_backbone'):])] = region_clip[key]
        #         elif key.startswith('lang_encoder'):
        #             region_clip[[key[len('lang_encoder') + 1:]]] = region_clip[key]
        #
        #     clip_model_w_adapter.load_state_dict(region_clip , strict=False)
        #     print(f"Load weight of pretrained CLIP from {args.pretrain_clip}")

        ao_pair = [(ACT_TO_ING[d['action']], d['object']) for d in HICO_INTERACTIONS]
        text_inputs = torch.cat(
            [clip.tokenize("a picture of person {} {}".format(a, o.replace('_', ' '))) for a, o in ao_pair]).to(device)
        text_features = clip_model_w_adapter.encode_text(text_inputs)   
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.to(device)    # 600,768

    model = PViC(
        (detr, args.detector), postprocessors['bbox'],
        feature_head=feature_head,
        ho_matcher=ho_matcher,
        triplet_decoder=triplet_decoder,
        num_verbs=args.num_verbs,
        repr_size=args.repr_dim,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        raw_lambda=args.raw_lambda,
        clip_model=clip_model_w_adapter,
        preprocess=preprocess,
        text_features=text_features
    )
    return model

class Adapter_3(nn.Module):
    # c_in 是特征的维度，reduction是bottleneck的倍数，论文中实验得到4是比较好的
    def __init__(self, c_in, reduction=4):
        super(Adapter_3, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in, c_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class Adapter(nn.Module):
    # c_in 是特征的维度，reduction是bottleneck的倍数，论文中实验得到4是比较好的
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in*reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in*reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class Adapter_1(nn.Module):
    # c_in 是特征的维度，reduction是bottleneck的倍数，论文中实验得到4是比较好的
    def __init__(self, c_in, reduction=4):
        super(Adapter_1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in, bias=False),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class My_Model_res(nn.Module):
    def __init__(self, clip_model, reduction=4, ratio=0.2):
        super().__init__()
        c_in = clip_model.text_projection.shape[1]
        self.visual_adapter = Adapter(c_in, reduction)
        self.language_adapter = Adapter(c_in, reduction)
        self.clip = clip_model
        self.ratio = ratio
        self.logit_scale = clip_model.logit_scale

    def encode_image(self, x):
        image_embedding = self.clip.encode_image(x)[:, 0]
        adpter_embedding = self.visual_adapter(image_embedding)
        image_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * image_embedding
        return image_embedding

    def encode_text(self, x):
        text_embedding = self.clip.encode_text(x)
        adpter_embedding = self.language_adapter(text_embedding)
        text_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * text_embedding
        return text_embedding

    def forward(self, image, text):
        image_embedding = self.encode_image(image)[:,0]
        adpter_embedding = self.visual_adapter(image_embedding)
        image_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * image_embedding

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        text_embedding = self.clip.encode_text(text)
        adpter_embedding = self.language_adapter(text_embedding)
        text_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * text_embedding
        # text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_embedding @ text_embedding.t()

        return logits

class My_Model(nn.Module):
    def __init__(self, clip_model, reduction=4, ratio=0.2):
        super().__init__()
        c_in = clip_model.text_projection.shape[1]
        self.visual_adapter = Adapter(c_in, reduction)
        self.language_adapter = Adapter(c_in, reduction)
        self.clip = clip_model
        self.logit_scale = clip_model.logit_scale
        # self.ratio = ratio

    def encode_image(self, x):
        return self.visual_adapter(self.clip.encode_image(x).float())
        # return self.clip.encode_image(x)[:,0]

    def encode_text(self, x):
        return self.language_adapter(self.clip.encode_text(x).float())  # x:600,77-->600,512
        # return self.clip.encode_text(x)

    def forward(self, image, text):
        image_embedding = self.encode_image(image)[:, 0]
        image_embedding = self.visual_adapter(image_embedding)
        # adpter_embedding = self.visual_adapter(image_embedding)
        # image_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * image_embedding

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        text_embedding = self.clip.encode_text(text)
        text_embedding = self.language_adapter(text_embedding)
        # adpter_embedding = self.language_adapter(text_embedding)
        # text_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * text_embedding
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_embedding @ text_embedding.t()

        return logits


class My_Model_Trans(nn.Module):
    def __init__(self, clip_model, reduction=4, ratio=0.2):
        super().__init__()
        c_in = clip_model.text_projection.shape[1]
        self.visual_adapter = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=4), num_layers=2)
        self.language_adapter = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=4), num_layers=2)
        self.clip = clip_model
        self.logit_scale = clip_model.logit_scale
        # self.ratio = ratio

    def encode_image(self, x):
        clip_embedding = self.clip.encode_image(x)[:,0]
        return self.visual_adapter(clip_embedding.unsqueeze(0).float()).squeeze(0)

    def encode_text(self, x):
        clip_embedding = self.clip.encode_text(x)
        return self.language_adapter(clip_embedding.unsqueeze(0).float()).squeeze(0)
    def forward(self, image, text):
        image_embedding = self.encode_image(image)
        # adpter_embedding = self.visual_adapter(image_embedding)
        # image_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * image_embedding
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        text_embedding = self.encode_text(text)
        # adpter_embedding = self.language_adapter(text_embedding)
        # text_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * text_embedding
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_embedding @ text_embedding.t()

        return logits

class My_Model_1_1(nn.Module):
    def __init__(self, clip_model, reduction=4, ratio=0.2):
        super().__init__()
        c_in = clip_model.text_projection.shape[1]
        self.visual_adapter = Adapter_1(c_in, reduction)
        self.language_adapter = Adapter_1(c_in, reduction)
        self.clip = clip_model
        self.logit_scale = clip_model.logit_scale
        # self.ratio = ratio

    def encode_image(self, x):
        return self.visual_adapter(self.clip.encode_image(x)[:,0])

    def encode_text(self, x):
        return self.language_adapter(self.clip.encode_text(x))


    def forward(self, image, text):
        image_embedding = self.encode_image(image)[:, 0]
        image_embedding = self.visual_adapter(image_embedding)
        # adpter_embedding = self.visual_adapter(image_embedding)
        # image_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * image_embedding

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        text_embedding = self.clip.encode_text(text)
        text_embedding = self.language_adapter(text_embedding)
        # adpter_embedding = self.language_adapter(text_embedding)
        # text_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * text_embedding
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_embedding @ text_embedding.t()

        return logits