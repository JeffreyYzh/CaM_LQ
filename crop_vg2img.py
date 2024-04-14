import json
import os
import torch
import argparse
from PIL import Image

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

def main(args):
    anno_dir = args.save_anno_dir

    with open(anno_dir, 'r') as f:
        annotations = json.load(f)

    save_img_dir = args.save_img_dir
    save_anno_dir = args.save_anno_dir


    target_idx = []
    save_annotations = []
    for anno in annotations:
        filename = anno['file_name']
        orig_image = Image.open(filename).convert('RGB')

        base_name = os.path.basename(filename)
        # 去掉文件后缀
        filename = os.path.splitext(base_name)[0]
        h_bbox = torch.as_tensor(anno['sub_box'], dtype=torch.float32).reshape(-1, 4)
        o_bbox = torch.as_tensor(anno['obj_box'], dtype=torch.float32).reshape(-1, 4)
        lt = torch.min(h_bbox[:, :2], o_bbox[:, :2])
        rb = torch.max(h_bbox[:, 2:], o_bbox[:, 2:])

        union_boxes = torch.cat((lt, rb), dim=1)

        for idx, (union_box, lab) in enumerate(zip(union_boxes,anno['hoi'])):
            x, y, x2, y2 = union_box.numpy()
            region = orig_image.crop((x, y, x2, y2))
            region = expand2square(region, (0, 0, 0))
            save_annotations.append(dict(
                file_name='{}_{}.jpg'.format(filename, idx),
                hoi=lab
            ))
            region.save(os.path.join(save_img_dir, '{}_{}.jpg'.format(filename, idx),), 'JPEG')

    with open(os.path.join(save_anno_dir, 'vg_hoi_annotation.json'), 'w') as f:
        json.dump(save_annotations, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--save_img_dir', default='/mnt/bn/motor-cv-yzh/acm/crop_img/hicodet/all_image', type=str)
    parser.add_argument('--save_anno_dir', default='./crop_img_RN50x16/vg/unionbox_crop_image_w_hoi.json', type=str)
    args = parser.parse_args()
    main(args)