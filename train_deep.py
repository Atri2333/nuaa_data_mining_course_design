# 深度学习方法
# commit: try tensorboard

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
import os

def train(model, num_epoch, train_iter, valid_iter, learning_rate, weight_decay, criterion, device, save_path, count=0):
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)

    best_acc = 0.0

    writer = SummaryWriter(log_dir="./result_dl")

    for epoch in range(num_epoch):
        model.train()

        i = 0
        train_loss = []
        train_accs = []

        for batch in tqdm(train_iter):
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            if i % 500 == 0:
                pass
            i = i + 1

            acc = (logits.argmax(dim=-1) == labels).float().mean()

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
            torch.save(model.state_dict(), os.path.join(save_path, f"best_acc{count}.pth"))
            print(f"save model with acc {best_acc:.3f} in epoch {epoch + 1}")

        writer.add_scalar("dl_acc", valid_acc, epoch)
        writer.add_scalar("dl_loss", valid_loss, epoch)
    
    torch.save(model.state_dict(), os.path.join(save_path, f"train_deep{count}.pth"))