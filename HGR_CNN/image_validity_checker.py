from PIL import Image
import os

current_dir_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir_path = os.path.join(current_dir_path, os.pardir, "dataset","mask2")

counter = 0
imgs = []
for filename in os.listdir(dataset_dir_path):
    counter = counter + 1
    name = os.path.join(dataset_dir_path,filename)
    im = Image.open(name)
    im.load()
    imgs.append(im)
    im.close()
    if counter%1000 == 0:
        print(counter)



