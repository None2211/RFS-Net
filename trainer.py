import torch
from tqdm import tqdm
from loss import BatchDiceLoss,average_dice_coefficient
import torch.nn as nn
import wandb



def train_step(model,data_loader,optimizer,device):
    model.train()

    total_seg_loss = 0.0

    total_loss = 0.0


    criterion_seg = BatchDiceLoss()
    criterion_seg_cls = nn.BCEWithLogitsLoss()
    progress_bar = tqdm(data_loader)
    for batch in progress_bar:

        img = batch['image'].to(dtype=torch.float32).to(device)
        superpixel_image =  batch['superpixel_image'].to(dtype=torch.float32).to(device)
        mask = batch['mask'].to(dtype=torch.float32).to(device)
        assert torch.all((mask == 0) | (mask == 1)), 'Mask contains values other than 0 and 1'



        output_seg = model(img,superpixel_image)
        #print('output_seg',output_seg.shape)
        loss_seg1 = criterion_seg(output_seg,mask)
        loss_seg2 = criterion_seg_cls(output_seg,mask)
        loss_seg = 0.6 * loss_seg1 + 0.4 * loss_seg2

        total_loss_back = loss_seg
        total_seg_loss += loss_seg.item()
        total_loss +=  loss_seg.item()
        # optimize #
        optimizer.zero_grad()
        total_loss_back.backward()
        optimizer.step()

        progress_bar.set_description(f"seg_loss:{loss_seg.item():},"
                                     f"total_loss:{total_loss_back.item()}")

    avg_seg_loss = total_seg_loss / len(data_loader)
    avg_loss = total_loss / len(data_loader)

    wandb.log({"seg_loss":avg_seg_loss,"total_loss":avg_loss})
    print("total loss:",total_loss)


def validation_step(model,data_loader, device):

    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    criterion_seg = BatchDiceLoss()
    criterion_seg_cls = nn.BCEWithLogitsLoss()

    with torch.no_grad():

        total_dice = 0.0


        progress_bar = tqdm(data_loader)

        for batch in progress_bar:
            img = batch['image'].to(dtype=torch.float32).to(device)
            superpixel_image =  batch['superpixel_image'].to(dtype=torch.float32).to(device)
            mask = batch['mask'].to(dtype=torch.float32).to(device)
            assert torch.all((mask == 0) | (mask == 1)), 'Mask contains values other than 0 and 1'

            output_seg = model(img,superpixel_image)
            loss_seg1 = criterion_seg(output_seg, mask)
            loss_seg2 = criterion_seg_cls(output_seg, mask)
            loss_seg = 0.6 * loss_seg1 + 0.4 * loss_seg2

            total_seg_loss += loss_seg.item()
            total_loss += loss_seg.item()

            preds = (torch.sigmoid(output_seg) > 0.5).float()
            dice = average_dice_coefficient(preds,mask)
            total_dice += dice



            progress_bar.set_postfix({
                'Dice': dice
            })
    avg_dice = total_dice / len(data_loader)

    avg_seg_loss = total_seg_loss / len(data_loader)

    avg_total_loss = total_loss / len(data_loader)

    wandb.log({
        "Validation dice score":avg_dice,
        "Validation seg_loss":avg_seg_loss,
        "Validation total_loss": avg_total_loss
    })

    print("Validation dice score:", avg_dice)
    print("Validation_total loss:", avg_total_loss)

    return avg_dice




















