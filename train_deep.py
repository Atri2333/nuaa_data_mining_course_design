# 深度学习方法
# commit: try tensorboard

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import numpy as np

from model.utils import *

def train(model, num_epoch, train_iter, valid_iter, learning_rate, weight_decay, criterion, device, save_path, mixup=False, cutmix=False, count=0, type=None):
    assert(mixup == False or cutmix == False)

    print(f"training on device: {device}, mixup: {mixup}, cutmix: {cutmix}, type: {type}")

    if mixup:
        print("mixup")
    else:
        print("fuck mixup")
    if cutmix:
        print("cutmix")
    else:
        print("fuck cutmix")

    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    best_acc = 0.0

    writer = SummaryWriter(log_dir=f"./result_dl_{type}_{mixup}_{cutmix}")

    for epoch in range(num_epoch):
        model.train()

        i = 0
        train_loss = []
        train_accs = []

        for batch in tqdm(train_iter):
            
            # if mixup:
            #     print("mixup")
            # else:
            #     print("fuck mixup")
            # if cutmix:
            #     print("cutmix")
            # else:
            #     print("fuck cutmix")

            if mixup == False and cutmix == False:
                imgs, labels = batch
                imgs, labels = imgs.to(device), labels.to(device)

                logits = model(imgs)

                loss = criterion(logits, labels)
            elif mixup:
                lam = np.random.beta(1., 1.)
                imgs, labels= batch
                imgs, labels = imgs.to(device), labels.to(device)
                batch_size = imgs.size()[0]
                index = torch.randperm(batch_size).to(device)
                mixed_x = lam * imgs + (1 - lam) * imgs[index, :]
                y_a, y_b = labels, labels[index]
                
                logits = model(mixed_x)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            elif cutmix:
                lam = np.random.beta(1., 1.)
                imgs, labels= batch
                imgs, labels = imgs.to(device), labels.to(device)
                batch_size = imgs.size()[0]
                index = torch.randperm(batch_size).to(device)

                bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                y_a, y_b = labels, labels[index]

                logits = model(imgs)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                print("Oh WTF???")


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % 500 == 0:
                pass
            i = i + 1

            if mixup == False and cutmix == False:
                acc = (logits.argmax(dim=-1) == labels).float().mean()
            else:
                acc = lam * (logits.argmax(dim=-1) == y_a).float().mean() + (1 - lam) * (logits.argmax(dim=-1) == y_b).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)
        
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        print(f"[Train | {epoch + 1: 03d}/{num_epoch: 03d}] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        model.eval()

        valid_loss = []
        valid_accs = []

        for batch in tqdm(valid_iter):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                logits = model(imgs)
            
            loss = criterion(logits, labels)

            acc = (logits.argmax(dim=-1) == labels).float().mean()

            valid_loss.append(loss.item())
            valid_accs.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        print(f"[Valid | {epoch + 1: 03d}/{num_epoch: 03d}] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(save_path, f"best_acc{count}_{type}_{mixup}_{cutmix}.pth"))
            print(f"save model with acc {best_acc:.3f} in epoch {epoch + 1}")

        writer.add_scalar(f"dl_acc_{count}_{type}_{mixup}_{cutmix}", valid_acc, epoch)
        writer.add_scalar(f"dl_loss_{count}_{type}_{mixup}_{cutmix}", valid_loss, epoch)
    
    torch.save(model.state_dict(), os.path.join(save_path, f"train_deep{count}_{type}_{mixup}_{cutmix}.pth"))