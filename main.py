import argparse
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import train_deep
from dl_model import *
from data import *
from utils import *
import knn, dummy

num_classes = 176
save_path = "submission.csv"

def predict(model, test_iter, device):
    model = model.to(device)
    predictions = []
    for batch in tqdm(test_iter):
        imgs = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs)
        predictions.append(logits.argmax(dim=-1).cpu().numpy().tolist())
    
    preds = []
    for i in predictions:
        preds.append(num_to_class[i])
        
    return predictions, preds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for training")
    parser.add_argument("--model_type", default="resnet34", type=str, help="single model or ensemble")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate if necessary")
    parser.add_argument("--wd", default=1e-3, type=float, help="weight decay if necessary")
    parser.add_argument("--num_epoch", default=50, type=int, help="counts of iteration")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size if necessary")
    parser.add_argument("--pretrained", type=str, default=True, help="Models were pretrained")
    parser.add_argument("--mode", default="train", type=str, help="train or predict")

    args = parser.parse_args()

    
    models = []
    if args.mode == "train":
        if args.model_type == "resnet34":
            models.append(res_model_34(num_classes, frozen=True, pretrained=args.pretrained))
        elif args.model_type == "resnext50":
            models.append(resnext_model_50(num_classes, frozen=True, pretrained=args.pretrained))

        train_trans = transforms.Compose([
            transforms.RandomResizedCrop((224,224), scale=(0.8, 1), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor()
        ])
        valid_trans = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])
        train_data = MyLeaveDataset(".", "train.csv", mode="train", valid_ratio=0.2, trans=train_trans)
        valid_data = MyLeaveDataset(".", "train.csv", mode="valid", valid_ratio=0.2, trans=valid_trans)
        test_data = MyLeaveDataset(".", "test.csv", mode="test", valid_ratio=0.114514, trans=valid_trans)

        train_iter = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
        valid_iter = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
        test_iter = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()

        if len(models) == 1:
            model = models[0]
            if isinstance(model, nn.Module):
                train_deep.train(model, args.num_epoch, train_iter, valid_iter, args.lr, args.wd, criterion, get_device(), ".")
    else:
        pass
