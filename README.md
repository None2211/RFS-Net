# RFS-Net: Region-Guided-Frequency-Spatial-Transformer-Network-for-Skin-Lesion-Segmentation
This repository contains the implementation of our paper "RFS-Net: Region-Guided-Frequency-Spatial-Transformer-Network-for-Skin-Lesion-Segmentation"
## Dataset
You can download the dataset from the official [ISIC](https://challenge.isic-archive.com/data/) website.
All datasets are organized and used for training, validation, and testing strictly following the official splits.
## Usage
1. Download datasets
2. Clone the repository, and download the pre-trained model [[Google Drive](https://drive.google.com/file/d/180JsahYjJkhnHEXbPHpet5BKo9pS7Bm0/view?usp=drive_link)], put them into ./ folder. The details of the training are in train.py file.
3. Use ```slic_extractor.py```  to extract the superpixel images corresponding to the input images
4. And then run the code：python train.py Note that the parameters and paths should be set beforehand
5. Once the training is complete, you can run the test.py to test your model. Run the code : python test.py
