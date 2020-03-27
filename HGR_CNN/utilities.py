import os
import re
import shutil

def purify_all_img_names(masks_path):
    all_image_names = os.listdir(masks_path)
    for img_name in all_image_names:
        regex_match = re.search(f"(depth_\d+_X.+_date.+)_x_\d+_y_\d+", img_name)
        if regex_match is not None:
            os.rename(os.path.join(masks_path, img_name), os.path.join(masks_path, regex_match.group(1)+".jpg"))

def create_mask_for_every_img(imgs_path):
    all_image_names = os.listdir(imgs_path)
    for img_name in all_image_names:
        regex_match = re.search(f"(depth_\d+_X.+_date.+)n", img_name)
        if regex_match is not None:
            mask_name = os.path.join(masks_path, regex_match.group(1)+".jpg")
            if not os.path.exists(mask_name):
                shutil.copy2(mask_name, os.path.join(masks_path, img_name))