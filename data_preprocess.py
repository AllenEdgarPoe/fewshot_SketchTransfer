from util import align_face
import os, PIL
import cv2
def rotate(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    im = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(img_path, im)

path = './data/train_chae2/sketch'
os.makedirs('./data/train_chae2/sketch_pro', exist_ok=True)
images = os.listdir(path)
for image in images:
    if image not in os.listdir('./data/train_chae2/sketch_pro'):
        image_path = os.path.join(path, image)
        rotate(image_path)
        try:
            im = align_face(image_path)
        except AssertionError:
            print(image_path)
        im.save(os.path.join('./data/train_chae2/sketch_pro', image))