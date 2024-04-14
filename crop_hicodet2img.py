from utils import DataFactory
from PIL import Image
import os
import json
import numpy as np
import clip
import torch
import argparse
import numpy as np

def main(args):
    dataset = 'hicodet'
    partition = 'train2015'
    dataroot = './hicodet'

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


    dataset = DataFactory(name=dataset, partition=partition, data_root=dataroot, setting=args.setting)

    annotations = np.array(dataset.dataset.annotations) # 38118
    annotations = annotations[dataset.dataset._idx]     # 37633
    filenames = np.array(dataset.dataset._filenames)    # 38118
    filenames = filenames[dataset.dataset._idx]         # 37633

    target_idx = [] # 117871
    for fn, ann in zip(filenames, annotations):
        hoi = ann['hoi']
        fn_id = fn.split(".")[0]
        for i in range(len(hoi)):
            target_idx.append(
                dict(file_name=fn_id + f"_{i}.jpg", hoi=hoi[i]))

    with open(args.save_anno_dir, "w") as f:
        json.dump(target_idx, f)

    # crop image 
    save_dir = args.save_dir

    target_idx = []

    for image, target in dataset:
        filename = target['file_name'][:-4]
        orig_img = target['org_img']
        union_boxes = target['union_boxes']
        regions = []
        for idx, union_box in enumerate(union_boxes):
            x, y, x2, y2 = union_box.numpy()
            region = orig_img.crop((x, y, x2, y2))
            region = expand2square(region, (0, 0, 0))
            region.save(os.path.join(save_dir, '{}_{}.jpg'.format(filename, idx)), 'JPEG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # Backbone
    parser.add_argument('--save_img_dir', default='/mnt/bn/motor-cv-yzh/acm/crop_img/hicodet/all_image', type=str)
    parser.add_argument('--save_anno_dir', default='/mnt/bn/motor-cv-yzh/acm/crop_img/hicodet/pretrain.json', type=str)
    parser.add_argument('--setting', default='uc0', type=str)

    args = parser.parse_args()
    main(args)