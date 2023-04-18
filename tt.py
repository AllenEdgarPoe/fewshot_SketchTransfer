import os
from PIL import Image

img_list = [f for f in os.listdir('./data/train_lim2/real_image') if f.endswith('.png')]

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

for img in img_list:

    if os.path.exists(os.path.join('./data/train_lim2/sketch_pro', img[:-3]+'jpg')):
        img1 = Image.open(os.path.join('./data/train_lim2/real_image', img))
        img2 = Image.open(os.path.join('./data/train_lim2/sketch_pro', img[:-3]+'jpg'))
        get_concat_h(img1, img2).save(os.path.join('./data/train_lim2', img))
