from ast import arg
import json
import clip
import torch
from hicodet.hico_text_label import hico_text_label
from hicodet.static_hico import ACT_IDX_TO_ACT_NAME, OBJ_IDX_TO_OBJ_NAME, HICO_INTERACTIONS
from vcoco.vcoco_text_label import vcoco_hoi_text_label
import wandb
import torch.distributed as dist
from loguru import logger
from clip.model import Transformer, LayerNorm, ModifiedResNet, VisionTransformer
from PIL import Image
import random
import clip
from torch.utils.data import DataLoader, DistributedSampler
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.optim as optim
import argparse
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from torch import nn
from pretrain_utils import custom_collate, save_on_master
import os
LABELS = list(hico_text_label.values())

os.environ["WANDB_API_KEY"] = 'fe7a5716372229e0694cacd97dbc4858bfc5b72c' # 将引号内的+替换成自己在wandb上的一串值
os.environ["WANDB_MODE"] = "offline"   # 离线  （此行代码不用修改）

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

#https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

# 先只用HICODET微调
class mydata(Dataset):
    def __init__(self, data_root, annotations, transforms):
        self.dataroot = data_root
        self.annotations = annotations
        self.image_root = os.path.join(data_root, 'all_image')
        self._hoi_text = list(hico_text_label.values()).extend(list(vcoco_hoi_text_label.values()))
        self.transforms = transforms

    def __getitem__(self, i):
        target = self.annotations[i]
        file_name = target['file_name']
        image = self.load_image(os.path.join(self.image_root, file_name))
        return self.transforms(image), target['hoi']
    def __len__(self):
        return len(self.annotations)

    def load_image(self, path: str) -> Image:
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')

class Adapter(nn.Module):
    # c_in 是特征的维度，reduction是bottleneck的倍数，论文中实验得到4是比较好的
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in * reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class My_Model(nn.Module):
    def __init__(self, clip_model, reduction=4, ratio=0.2):
        super().__init__()
        c_in = clip_model.text_projection.shape[1]
        self.visual_adapter = Adapter(c_in, reduction)
        self.language_adapter = Adapter(c_in, reduction)
        self.clip = clip_model
        self.ratio = ratio

    def encode_image(self, x):
        return self.clip.encode_image(x)

    def encode_text(self, x):
        return self.clip.encode_text(x)
    def forward(self, image, text):
        image_embedding = self.encode_image(image)[:,0]
        image_embedding = self.visual_adapter(image_embedding)
        adpter_embedding = self.visual_adapter(image_embedding)
        image_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * image_embedding

        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)

        text_embedding = self.clip.encode_text(text)
        text_embedding = self.ratio * self.language_adapter(text_embedding)
        adpter_embedding = self.language_adapter(text_embedding)
        text_embedding = self.ratio * adpter_embedding + (1 - self.ratio) * text_embedding
        # text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * image_embedding @ text_embedding.t()

        return logits

def main(args):
    init_distributed_mode(args)

    backbone = 'ViT-B/16'
    data_root = args.img_dir
    output_dir = args.output

    # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(backbone, device=device, jit=False)
    model = My_Model(clip_model)
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 创建数据集合
    random.seed(100)
    with open(args.anno_dir, 'r') as f:
        annotations = json.load(f)
    index = [i for i in range(len(annotations))]
    random.shuffle(index)
    annotations = np.array(annotations)[index]

    train_anno = annotations[:-10000]
    test_anno = annotations[-10000:]
    train_dataset = mydata(data_root, annotations=train_anno, transforms=preprocess)
    test_dataset = mydata(data_root, annotations=test_anno, transforms=preprocess)

    if args.distributed:
        sampler_train = DistributedSampler(train_dataset)
        sampler_val = DistributedSampler(test_dataset, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(test_dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(train_dataset, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(test_dataset, args.batch_size, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    # freeze image encoder
    # 创建优化器
    for n, p in model.named_parameters():
        if 'adapter' in n or 'text_projection' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    scheduler = lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Model initialized from {args.resume}")

    # 设置训练和记录参数
    best_performance = 0

    wandb.init(
                project = 'pretrain_clip',
                config = args,
                name = 'debug',
                resume = True if args.resume else False,
                )

    input_text = list(hico_text_label.values())
    input_token = clip.tokenize(input_text)

    if args.eval:
        with torch.no_grad():
            model.eval()
            precision = 0
            for images, targets in data_loader_val:
                images = images.to(device)
                targets = targets.to(device)
                input_token = input_token.to(device)

                image_features = model.encode_image(images)[:, 0]
                text_features = model.encode_text(input_token)

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()

                # shape = [global_batch_size, global_batch_size]
                pred = torch.nn.Softmax(dim=-1)(logits_per_image)

                precision += sum(pred.argmax(-1) == targets).item()

            precision = precision / len(test_dataset)
            print(f"Test precision is: {precision}")

    else:
        # add your own code to track the training progress.
        for epoch in range(args.epoch):
            model.train()
            total_loss = 0
            for idx, (images, targets) in enumerate(data_loader_train):
                images = images.to(device)
                targets = targets.to(device)
                input_token = input_token.to(device)
                logits = model(images, input_token)
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item()

                loss.backward()
                if device == "cpu":
                    optimizer.step()
                else:
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

                optimizer.zero_grad()

                if idx % 20 == 0:
                    logger.info('epoch{} {}iter:train loss:{}'.format(epoch, idx, loss.item()))

            epoch_loss = total_loss / len(train_dataset)

            checkpoint_path = os.path.join(output_dir, 'last.pth')
            save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
            print(f"weights_{epoch} saved")
            with torch.no_grad():
                model.eval()
                precision = 0
                for images, targets in data_loader_val:
                    images = images.to(device)
                    targets = targets.to(device)
                    input_token = input_token.to(device)
                    logits = model(images, input_token)
                    pred = torch.nn.Softmax(dim=-1)(logits)

                    precision += sum(pred.argmax(-1)==targets).item()

                precision = precision / len(test_dataset)
                print(f"Epoch{epoch} Train Loss:{epoch_loss}, precision:{precision}")

                if precision > best_performance:
                    best_performance = precision
                    checkpoint_path = os.path.join(output_dir, 'best.pth')
                    save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)
                    print(f"Best model has been saved")

            wandb.log({'epoch_loss': epoch_loss,
                       'epoch':epoch,
                       'precision':precision})

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoch', default=50, type=float)
    parser.add_argument('--batch_size', default=2000, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--data_root', default='./crop_img_ViT-B-32', help='resume from checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--anno_dir', default='')
    parser.add_argument('--img_dir', default='')
    parser.add_argument('--resume', default='./pretrained_clip/ViT-B16/image-text-adapter22-res/best.pth', help='resume from checkpoint')
    parser.add_argument('--output', default='./pretrained_clip/ViT-B16/image-text-adapter22-res', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    args = parser.parse_args()
    main(args)