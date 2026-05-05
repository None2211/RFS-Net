import os
import torch
from torch.utils.data import Dataset
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import numpy as np
from random_transforms import RandomTransforms

class Multimodel_dataset(Dataset):
    def __init__(self, img_dir, superpixel_dir, mask_dir, excel_path, transforms=None,scale: float = 1.0):
        """
        img_dir: img_folder
        superpixel_dir: superpixel_folder
        mask_dir: mask_folder
        excel_path: cls
        transform:
        """

        self.img_dir = img_dir
        self.superpixel_dir = superpixel_dir
        self.mask_dir = mask_dir
        self.label_df = pd.read_csv(excel_path)
        self.label_df.set_index("filename", inplace=True)
        self.scale = scale
        self.transforms = transforms

    def __len__(self):
        return len(self.label_df)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            if img.ndim == 2:
                mask = np.where(img == 255, 1, 0)
                mask = np.expand_dims(mask, axis=0)

            else:
                assert 'dimension more than one'

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):

            image_id = self.label_df.iloc[idx].name
            cls_label = self.label_df.iloc[idx]['label']
            #label = torch.tensor([cls_label], dtype=torch.float32)

            # 加载图像和对应的超像素图像
            img_path = os.path.join(self.img_dir,f"{image_id}")
            img = Image.open(img_path).convert("RGB")
            #img = self.preprocess(img,self.scale,is_mask=False)
            super_id = image_id.replace('.png', '', 1)

            superpixel_path = os.path.join(self.superpixel_dir, f"{super_id}.png")
            superpixel_img = Image.open(superpixel_path).convert("RGB")
            #superpixel_img = self.preprocess(superpixel_img, self.scale, is_mask=False)

            # 加载掩码
            mask_id = image_id.replace('.png', '', 1)
            mask_path = os.path.join(self.mask_dir, f"{mask_id}_segmentation.png")
            mask = Image.open(mask_path).convert("L")
            #mask = self.preprocess(mask,self.scale,is_mask=True)# L模式表示灰度图

            # 如果存在转换，则应用它
            if self.transforms:
                img, superpixel_img, mask = self.transforms(img, superpixel_img, mask)

            img = self.preprocess(img, self.scale, is_mask=False)
            superpixel_img = self.preprocess(superpixel_img, self.scale, is_mask=False)
            mask = self.preprocess(mask, self.scale, is_mask=True)

            return {
                'image_path': img_path,
                'superpixel_path': superpixel_path,
                'mask_path': mask_path,
                'image': torch.as_tensor(np.array(img), dtype=torch.float32).contiguous(),
                'superpixel_image': torch.as_tensor(np.array(superpixel_img), dtype=torch.float32).contiguous(),
                'mask': torch.as_tensor(np.array(mask), dtype=torch.float32).contiguous()
            }

if __name__ == '__main__':
    transform = RandomTransforms()
    dataset = Multimodel_dataset(img_dir=r'your_url',
                            superpixel_dir=r'your_url',
                            mask_dir=r'your_url',
                            excel_path=r'your_url',
                            transforms=transform)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for data in tqdm(dataloader):
        img_path = data['image_path']
        mask_path = data['mask_path']
        img = data['image']
        mask = data['mask']
        print(mask.shape)
        valid_mask = (mask == 0) | (mask == 1)
        assert valid_mask.all(), 'Mask contains values other than 0 and 255'
        unique_values = torch.unique(mask)
        #print(unique_values)



        #label = data['label']
        #print(label.shape)

        #print(img.shape)
        #print(label.shape)
