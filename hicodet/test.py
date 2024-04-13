import os
from PIL import Image
from static_hico import HICO_INTERACTIONS
import clip
image_dir = '../crop_img/vcoco/image'
image_list = os.listdir(image_dir)[0]
picture = Image.open(os.path.join(image_dir, image_list))
# model, preprocess = clip.load('RN50x16', device='cuda:0')   #   3,384,384
# model, preprocess = clip.load('RN50x4', device='cuda:0')  # 3,288, 288
# model, preprocess = clip.load('RN50x64', device='cuda:0')   # 3,448,448 #
# model, preprocess = clip.load('ViT-B/32', device='cuda:0')  # 3,224,224
# model, preprocess = clip.load('ViT-B/16', device='cuda:0')  # 3,224,224
model, preprocess = clip.load('ViT-L/14', device='cuda:0')  # 3,224,224
input = preprocess(picture) # 3,224,224

a = 1