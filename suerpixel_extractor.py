import cv2
import os
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb


input_dir = r"your_url"
output_dir = r"your_url"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):

        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file) 

        img = cv2.imread(in_path)
        if img is None:
            print(f"fail: {file}")
            continue

        img_rgb = img[:, :, ::-1]  


        segments = slic(img_rgb, n_segments=200, compactness=20, sigma=1, start_label=1)


        superpixel_img = label2rgb(segments, img_rgb, kind='avg')


        superpixel_img_bgr = (superpixel_img * 255).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(out_path, superpixel_img_bgr)


print("completed！")
