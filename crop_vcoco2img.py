import torch
from utils import DataFactory
from PIL import Image
import os
import json
import numpy as np
import argparse
import clip

def main():
    # len1 = len(os.listdir('crop_img/vcoco/image'))    # 117871这是把所有的crop_image都放在里面了
    # with open(os.path.join('crop_img/vcoco', 'vcoco_annotation.json'), 'r') as f:
    #     anno = json.load(f) # 长度为4969的list,以每张图片为单位
    #     all_interactions = [act_obj for ann in anno for act_obj in ann['interaction']]  # 13817
    #     with open(os.path.join('crop_img/vcoco', 'all_annotation.json'), 'w') as f:
    #         json.dump(all_interactions, f)

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

    # 用clip预处理存下来
    backbone = 'ViT-B-16'
    clip_model, preprocess = clip.load('./ckpt/ViT-B-16.pt')

    dataset = 'vcoco'
    partition = 'trainval'
    dataroot = './vcoco'

    save_dir = './crop_img' + '_' + backbone
    partition_path = 'vcoco'
    image_path = 'image'
    anno_path = 'annotation.json'

    dataset = DataFactory(name=dataset, partition=partition, data_root=dataroot)

    annotations = np.array(dataset.dataset.annotations)
    annotations = annotations[dataset.dataset._keep]
    filenames = np.array([anno['file_name'] for anno in dataset.dataset.annotations])
    filenames = filenames[dataset.dataset._keep]

    # dir = './crop_img/vcoco/vcoco_annotation_hoi.json'
    # with open(dir, 'r') as f:
    #     fn_hoi_annotations = json.load(f)

    # inter_objs = []
    # for anno in annotations:
    #     tmp = []
    #     for act, obj in zip(anno['actions'], anno['objects']):
    #         tmp.append((act, obj))
    #     inter_objs.append(tmp)

    # root_dir = os.listdir(os.path.join(save_dir, partition_path, image_path))

    for image, target in dataset:
        filename = target['file_name'][:-4]

        # if '{}.pth'.format(filename) in root_dir:
        #     continue

        orig_img = target['org_img']
        union_boxes = target['union_boxes']
        regions = []
        for idx, union_box in enumerate(union_boxes):
            x, y, x2, y2 = union_box.numpy()
            region = orig_img.crop((x, y, x2, y2))
            region = expand2square(region, (0, 0, 0))
            region = preprocess(region)
            regions.append(region)

        regions = torch.stack(regions)
        save_region_dir = os.path.join(save_dir, partition_path, image_path, '{}.pth'.format(filename))
        torch.save(regions, save_region_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # Backbone
    parser.add_argument('--save_img_dir', default='/mnt/bn/motor-cv-yzh/acm/crop_img/hicodet/all_image', type=str)
    parser.add_argument('--save_anno_dir', default='/mnt/bn/motor-cv-yzh/acm/crop_img/hicodet/pretrain.json', type=str)
    args = parser.parse_args()
    main(args)