from util import *
import os, PIL
path = './data/train/sketch2'
os.makedirs('./data/train/sketch_pro2', exist_ok=True)
images = os.listdir(path)
for image in images:
    if image not in os.listdir('./data/train/sketch_pro2'):
        image_path = os.path.join(path, image)
        im = align_face(image_path)
        im.save(os.path.join('./data/train/sketch_pro2', image))