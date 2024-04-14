"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Microsoft Research Asia
"""

import os
import time
import clip
import torch
import pickle
import numpy as np
import scipy.io as sio
from hicodet.static_hico import UO_HOI_IDX,UC_HOI_IDX,UA_HOI_IDX
try:
    import wandb
except ImportError:
    pass

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet as HICODet
from hicodet.hicodet_ua import HICODet as HICODet_UA
from hicodet.hicodet_uc import HICODet as HICODet_UC
from hicodet.hicodet_rfuc import HICODet as HICODet_RFUC
from hicodet.hicodet_nfuc import HICODet as HICODet_NFUC
from hicodet.hicodet_uo import HICODet as HICODet_UO
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

from ops import recover_boxes
from detr.datasets import transforms as T

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def custom_collate(batch):
    images = []
    targets = []
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root, setting='default', tensor_root='./crop_img_ViT-B-16/hicodet/image'):
        _, self.preprocess = clip.load('ViT-L/14')
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            if partition == 'train2015':
                self.training = True
                self.root_dir = tensor_root
            else:
                self.training = False
                self.root_dir = './crop_img_ViT-B-16/vcoco/image'
            self.setting = setting
            if setting == 'UA':
                self.dataset = HICODet_UA(
                    root=os.path.join(data_root, "hico_20160224_det/images", partition),
                    anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            elif setting == 'uc0':
                self.dataset = HICODet_UC(
                    root=os.path.join(data_root, "hico_20160224_det/images", partition),
                    anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            elif setting == 'rare_first':
                self.dataset = HICODet_RFUC(
                    root=os.path.join(data_root, "hico_20160224_det/images", partition),
                    anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            elif setting == 'non_rare_first':
                self.dataset = HICODet_NFUC(
                    root=os.path.join(data_root, "hico_20160224_det/images", partition),
                    anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            elif setting == 'UO':
                self.dataset = HICODet_UO(
                    root=os.path.join(data_root, "hico_20160224_det/images", partition),
                    anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )
            else:
                self.dataset = HICODet(
                    root=os.path.join(data_root, "hico_20160224_det/images", partition),
                    anno_file=os.path.join(data_root, f"instances_{partition}.json"),
                    target_transform=pocket.ops.ToTensor(input_format='dict')
                )

        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            self.training = False
            if partition == 'trainval' or partition == 'train':
                self.training = True
                self.root_dir = './crop_img_ViT-B-16/vcoco/image'

            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, f"instances_vcoco_{partition}.json"),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # Prepare dataset transforms
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ])
                ), normalize,
        ])
        else:
            self.transforms = T.Compose([
                T.RandomResize([800], max_size=1333),
                normalize,
            ])

        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        (image, target), org_img, idx = self.dataset[i]
        org_anno = self.dataset._anno[idx]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
            # target['org_img'] = org_img
        else:
            target['labels'] = target['actions']
            target['verb'] = target['actions']
            target['object'] = target.pop('objects')

        image, target = self.transforms(image, target)
        org_sub_boxes, org_obj_boxes = org_anno['boxes_h'], org_anno['boxes_o']
        h_bbox = torch.as_tensor(org_sub_boxes, dtype=torch.float32).reshape(-1, 4)
        o_bbox = torch.as_tensor(org_obj_boxes, dtype=torch.float32).reshape(-1, 4)
        lt = torch.min(h_bbox[:, :2], o_bbox[:, :2])
        rb = torch.max(h_bbox[:, 2:], o_bbox[:, 2:])
        target['union_boxes'] = torch.cat((lt, rb), dim=1)

        if self.name == 'hicodet':
            target['file_name'] = self.dataset._filenames[idx]
        else:
            target['file_name'] = self.dataset.annotations[idx]['file_name']

        if self.training: # 产生数据集的时候注释掉
            target['clip_embed'] = self.preprocess(org_img)  # CLIP 整体distill

            if 'seen_target' in target.keys():
                seen_idx = target['seen_target']
                target['union_tensors'] = torch.load(os.path.join(self.root_dir, f"{target['file_name'][:-4]}.pth"))[seen_idx]
            else:
                target['union_tensors'] = torch.load(os.path.join(self.root_dir, f"{target['file_name'][:-4]}.pth"))
        return image, target

class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_dataloader, test_dataloader, config):
        super().__init__(
            net, None, train_dataloader,
            print_interval=config.print_interval,
            cache_dir=config.output_dir,
            find_unused_parameters=True
        )
        self.config = config
        self.max_norm = config.clip_max_norm
        self.test_dataloader = test_dataloader
        self.loss_weight = dict(
            verb_loss=config.verb_loss,
            clip_loss=config.clip_loss
        )
    def _on_start(self):
        if self._train_loader.dataset.name == "hicodet":
            ap = self.test_hico()
            if self._rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = self.test_dataloader.dataset.dataset.rare
                non_rare = self.test_dataloader.dataset.dataset.non_rare

                perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )

                log_dict = {
                    "epochs": self._state.epoch, "mAP full": perf[0],
                    "mAP rare": perf[1], "mAP non_rare": perf[2]
                }

                if self.config.setting == 'UA':
                    ua = self.test_dataloader.dataset.dataset.ua
                    sa = self.test_dataloader.dataset.dataset.sa
                    print(
                        f"unseen_action: {ap[ua].mean().item():.4f}, seen_action: {ap[sa].mean().item():.4f}."
                    )
                    log_dict['unseen_action'] = ap[ua].mean().item()
                    log_dict['seen_action'] = ap[sa].mean().item()
                elif self.config.setting == 'uc0' or self.config.setting == 'rare_first' or self.config.setting == 'non_rare_first':
                    uc = self.test_dataloader.dataset.dataset.uc
                    sc = self.test_dataloader.dataset.dataset.sc
                    print(
                        f"unseen_composition: {ap[uc].mean().item():.4f}, seen_composition: {ap[sc].mean().item():.4f}."
                    )
                    log_dict['unseen_composition'] = ap[uc].mean().item()
                    log_dict['seen_compositionn'] = ap[sc].mean().item()

                elif self.config.setting == 'UO':
                    uo = self.test_dataloader.dataset.dataset.uo
                    so = self.test_dataloader.dataset.dataset.so
                    print(
                        f"unseen_object: {ap[uo].mean().item():.4f}, seen_object: {ap[so].mean().item():.4f}."
                    )
                    log_dict['unseen_object'] = ap[uo].mean().item()
                    log_dict['seen_object'] = ap[so].mean().item()

                self.best_perf = perf[0]
                wandb.init(project='pvic_exp',config=self.config)
                wandb.watch(self._state.net.module)
                wandb.define_metric("epochs")   # 定义需要记录的指标名称
                wandb.define_metric("mAP full", step_metric="epochs", summary="max")    # step_metric是横坐标的意思
                wandb.define_metric("mAP rare", step_metric="epochs", summary="max")
                wandb.define_metric("mAP non_rare", step_metric="epochs", summary="max")
                if self.config.setting == 'UA':
                    wandb.define_metric("unseen_action", step_metric="epochs", summary="max")   # 汇总指标显示该指标的最小值、最大值、平均值或最佳值
                    wandb.define_metric("seen_action", step_metric="epochs", summary="max")

                if self.config.setting == 'uc0' or self.config.setting == 'rare_first' or self.config.setting == 'non_rare_first':
                    wandb.define_metric("unseen_composition", step_metric="epochs", summary="max")   # 汇总指标显示该指标的最小值、最大值、平均值或最佳值
                    wandb.define_metric("seen_composition", step_metric="epochs", summary="max")

                if self.config.setting == 'UO':
                    wandb.define_metric("unseen_object", step_metric="epochs", summary="max")   # 汇总指标显示该指标的最小值、最大值、平均值或最佳值
                    wandb.define_metric("seen_object", step_metric="epochs", summary="max")

                wandb.define_metric("training_steps")
                wandb.define_metric("elapsed_time", step_metric="training_steps", summary="max")
                wandb.define_metric("loss", step_metric="training_steps", summary="min")

                wandb.log(log_dict)
        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(),]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                self.best_perf = perf[0]
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """
                wandb.init(config=self.config)

    def _on_end(self):
        if self._rank == 0:
            wandb.finish()

    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        # if loss_dict['verb_loss'].isnan() or loss_dict['clip_loss'].isnan():  仅用来加入CLIP损失
        #     raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss_dict[loss_type] * self.loss_weight[loss_type] for loss_type in loss_dict.keys())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()

    def _print_statistics(self):
        running_loss = self._state.running_loss.mean()
        t_data = self._state.t_data.sum() / self._world_size
        t_iter = self._state.t_iteration.sum() / self._world_size

        # Print stats in the master process
        if self._rank == 0:
            num_iter = len(self._train_loader)
            n_d = len(str(num_iter))
            print(
                "Epoch [{}/{}], Iter. [{}/{}], "
                "Loss: {:.4f}, "
                "Time[Data/Iter.]: [{:.2f}s/{:.2f}s]".format(
                self._state.epoch, self.epochs,
                str(self._state.iteration - num_iter * (self._state.epoch - 1)).zfill(n_d),
                num_iter, running_loss, t_data, t_iter
            ))
            wandb.log({
                "elapsed_time": (time.time() - self._dawn) / 3600,
                "training_steps": self._state.iteration,
                "loss": running_loss
            })
        self._state.t_iteration.reset()
        self._state.t_data.reset()
        self._state.running_loss.reset()

    def _on_end_epoch(self):
        if self._train_loader.dataset.name == "hicodet":
            ap = self.test_hico()
            if self._rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = self.test_dataloader.dataset.dataset.rare
                non_rare = self.test_dataloader.dataset.dataset.non_rare

                perf = [ap.mean().item(), ap[rare].mean().item(), ap[non_rare].mean().item()]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}, rare: {perf[1]:.4f}, none-rare: {perf[2]:.4f}."
                )

                log_dict = {
                    "epochs": self._state.epoch, "mAP full": perf[0],
                    "mAP rare": perf[1], "mAP non_rare": perf[2]
                }

                # ua = UA_HOI_IDX
                # sa = list(range(600))
                # for id in ua:
                #     sa.remove(id)
                #
                # print(f"unseen_action: {ap[ua].mean().item():.4f}, seen_action: {ap[sa].mean().item():.4f}.")
                #
                # uo = UO_HOI_IDX
                # so = list(range(600))
                # for id in uo:
                #     so.remove(id)
                # print(f"unseen_object: {ap[uo].mean().item():.4f}, seen_object: {ap[so].mean().item():.4f}.")
                #
                # uc = UC_HOI_IDX['rare_first']
                # sc = list(range(600))
                # for id in uc:
                #     sc.remove(id)
                # print(f"rare_first uc: {ap[uc].mean().item():.4f}, rare_first sc: {ap[sc].mean().item():.4f}.")
                #
                #
                # uc = UC_HOI_IDX['non_rare_first']
                # sc = list(range(600))
                # for id in uc:
                #     sc.remove(id)
                # print(f"non_rare_first uc: {ap[uc].mean().item():.4f}, non_rare_first sc: {ap[sc].mean().item():.4f}.")
                #
                # uc = UC_HOI_IDX['uc0']
                # sc = list(range(600))
                # for id in uc:
                #     sc.remove(id)
                # print(f"uc0 uc: {ap[uc].mean().item():.4f}, uc0 sc: {ap[sc].mean().item():.4f}.")


                if self.config.setting == 'UA':
                    ua = self.test_dataloader.dataset.dataset.ua
                    sa = self.test_dataloader.dataset.dataset.sa
                    print(
                        f"unseen_action: {ap[ua].mean().item():.4f}, seen_action: {ap[sa].mean().item():.4f}."
                    )
                    log_dict['unseen_action'] = ap[ua].mean().item()
                    log_dict['seen_action'] = ap[sa].mean().item()
                elif self.config.setting == 'uc0' or self.config.setting == 'rare_first' or self.config.setting == 'non_rare_first':
                    uc = self.test_dataloader.dataset.dataset.uc
                    sc = self.test_dataloader.dataset.dataset.sc
                    print(
                        f"unseen_composition: {ap[uc].mean().item():.4f}, seen_composition: {ap[sc].mean().item():.4f}."
                    )
                    log_dict['unseen_composition'] = ap[uc].mean().item()
                    log_dict['seen_composition'] = ap[sc].mean().item()
                elif self.config.setting == 'UO':
                    uo = self.test_dataloader.dataset.dataset.uo
                    so = self.test_dataloader.dataset.dataset.so
                    print(
                        f"unseen_object: {ap[uo].mean().item():.4f}, seen_object: {ap[so].mean().item():.4f}."
                    )
                    log_dict['unseen_object'] = ap[uo].mean().item()
                    log_dict['seen_object'] = ap[so].mean().item()

                wandb.log(log_dict)

        else:
            ap = self.test_vcoco()
            if self._rank == 0:
                perf = [ap.mean().item(),]
                print(
                    f"Epoch {self._state.epoch} =>\t"
                    f"mAP: {perf[0]:.4f}."
                )
                """
                NOTE wandb was not setup for V-COCO as the dataset was only used for evaluation
                """

        if self._rank == 0:
            # Save checkpoints
            checkpoint = {
                'iteration': self._state.iteration,
                'epoch': self._state.epoch,
                'performance': perf,
                'model_state_dict': self._state.net.module.state_dict(),
                'optim_state_dict': self._state.optimizer.state_dict(),
                'scaler_state_dict': self._state.scaler.state_dict()
            }
            if self._state.lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._state.lr_scheduler.state_dict()
            torch.save(checkpoint, os.path.join(self._cache_dir, "latest.pth"))
            if perf[0] > self.best_perf:
                self.best_perf = perf[0]
                torch.save(checkpoint, os.path.join(self._cache_dir, "best.pth"))
        if self._state.lr_scheduler is not None:
            self._state.lr_scheduler.step()

    @torch.no_grad()
    def test_hico(self):
        dataloader = self.test_dataloader
        net = self._state.net; net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))  # object和其可能产生交互的pair的对应关系

        if self._rank == 0:
            meter = DetectionAPMeter(
                600, nproc=1, algorithm='11P',
                num_gt=dataset.anno_interaction,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]


            scores_clt = []; preds_clt = []; labels_clt = []
            for output, target in zip(outputs, targets):
                # KO setting preprocess
                # keep = list(range(len(output['objects'])))
                # for i in range(len(output['objects'])):
                #     if output['objects'][i] not in target['object']:
                #         keep.remove(i)

                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                verbs = output['labels']
                objects = output['objects']
                interactions = conversion[objects, verbs]
                # Recover target box scale
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(interactions)
                labels_clt.append(labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, (image, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image,])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def test_vcoco(self):
        dataloader = self.test_dataloader
        net = self._state.net; net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)

        if self._rank == 0:
            meter = DetectionAPMeter(
                24, nproc=1, algorithm='11P',
                num_gt=dataset.num_instances,
            )
        for batch in tqdm(dataloader, disable=(self._world_size != 1)):
            inputs = pocket.ops.relocate_to_cuda(batch[:-1])
            outputs = net(*inputs)
            outputs = pocket.ops.relocate_to_cpu(outputs, ignore=True)
            targets = batch[-1]

            scores_clt = []; preds_clt = []; labels_clt = []
            for output, target in zip(outputs, targets):
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
                scores = output['scores']
                actions = output['labels']
                gt_bx_h = recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = recover_boxes(target['boxes_o'], target['size'])

                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_actions = actions.unique()
                for act_idx in unique_actions:
                    gt_idx = torch.nonzero(target['actions'] == act_idx).squeeze(1)
                    det_idx = torch.nonzero(actions == act_idx).squeeze(1)
                    if len(gt_idx):
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )

                scores_clt.append(scores)
                preds_clt.append(actions)
                labels_clt.append(labels)
            # Collate results into one tensor
            scores_clt = torch.cat(scores_clt)
            preds_clt = torch.cat(preds_clt)
            labels_clt = torch.cat(labels_clt)
            # Gather data from all processes
            scores_ddp = pocket.utils.all_gather(scores_clt)
            preds_ddp = pocket.utils.all_gather(preds_clt)
            labels_ddp = pocket.utils.all_gather(labels_clt)

            if self._rank == 0:
                meter.append(torch.cat(scores_ddp), torch.cat(preds_ddp), torch.cat(labels_ddp))

        if self._rank == 0:
            ap = meter.eval()
            return ap
        else:
            return -1

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, (image, target) in enumerate(tqdm(dataloader.dataset)):
            inputs = pocket.ops.relocate_to_cuda([image,])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(1)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)
