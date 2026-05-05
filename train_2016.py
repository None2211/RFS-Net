import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader

from loss import average_dice_coefficient,BatchDiceLoss
from random_transforms import  RandomTransforms
from dataset_2016 import Multimodel_dataset
from multi_arch import Multi_arch
from RFS-Net import csunet
from trainer import train_step,validation_step



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    transform = RandomTransforms()
    seed_everything(42)
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--epochs',default=50,type=int,help='Number of epochs')
    parser.add_argument('--batch_size',default=24,type=int,help='Batch size')
    parser.add_argument('--lr',default=0.001,type=float,help='Learning rate')
    parser.add_argument('--load_seg',default=None,type=str,help='load weight')

    args = parser.parse_args()
    train_img = r'your_url'
    train_super = r'your_url'
    train_cls = r'your_url'
    train_mask = r'your_url'

    val_img = r'your_url'
    val_cls = r'your_url'
    val_mask = r'your_url'
    val_super = r'your_url'

    train_dataset = Multimodel_dataset(img_dir=train_img, mask_dir=train_mask,excel_path=train_cls,superpixel_dir=train_super
                                       ,transforms=None)
    val_dataset = Multimodel_dataset(img_dir=val_img, mask_dir=val_mask,excel_path=val_cls,superpixel_dir=val_super,
                                     transforms=None)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = csunet(num_class=1,num_cls=3).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.1)

    best_score =  0.0
    if args.load_seg:
        model.load_state_dict(torch.load(args.load_seg),strict=False)
    print('Start training...')
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        train_step(model,train_dataloader,optimizer,device)

        avg_dice = validation_step(model, val_dataloader,device)

        combine = avg_dice

        scheduler.step(combine)

        if best_score < combine:
            best_score = combine
            torch.save(model.state_dict(),"best.pth")
            print(f"New best score: {best_score:.4f}")

        if (epoch + 1) % 10 == 0:
            save_path = f"model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path} for epoch {epoch + 1}")


if __name__ == '__main__':
    main()













