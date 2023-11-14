import argparse
from torchvision import transforms
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

import train_deep
from model.dl_model import *
from data import *
from model.utils import *
import model.knn, model.dummy, model.svm

num_classes = 176
save_path = "submission.csv"
csv_path = "train.csv"

def predict(model, test_iter, device, test_dir, save_dir):
    model = model.to(device)
    predictions = []
    model.eval()
    for batch in tqdm(test_iter):
        imgs = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            logits = model(imgs)
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    
    preds = []
    for i in predictions:
        preds.append(num_to_class[i])
        
    test_data = pd.read_csv(test_dir)
    test_data["label"] = pd.Series(preds)
    submission = pd.concat([test_data["image"], test_data["label"]], axis=1)
    submission.to_csv(save_dir, index=False)
    # return predictions, preds

def test_on_wholedata_single_model(model, data_iter, device):
    model = model.to(device)
    accs = []
    predictions = []
    for batch in tqdm(data_iter):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(imgs)
        predictions.append(logits.argmax(dim=-1).cpu().numpy().tolist())
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        accs.append(acc)
        
    acc = sum(accs) / len(accs)
    return predictions, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for training")
    parser.add_argument("--model_type", default="resnet34", type=str, help="single model or ensemble")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate if necessary")
    parser.add_argument("--wd", default=1e-3, type=float, help="weight decay if necessary")
    parser.add_argument("--num_epoch", default=30, type=int, help="counts of iteration")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size if necessary")
    parser.add_argument("--pretrained", action="store_true", help="Models were pretrained")
    parser.add_argument("--mixup", action="store_true", help="using mixup trick")
    parser.add_argument("--cutmix", action="store_true", help="using cutmix trick")
    parser.add_argument("--mode", default="train", type=str, help="train or predict")

    args = parser.parse_args()

    if args.model_type == "dummy":
        X, y = model.dummy.getDataSet(".", csv_path)
        model.dummy.Classify(X, y)
        exit()
    elif args.model_type == "knn4hog":
        model.knn.knn4hog()
        exit()
    elif args.model_type == "knn4sift":
        model.knn.knn4sift()
        exit()
    elif args.model_type == "svm4sift":
        model.svm.svm4sift()
        exit()
    elif args.model_type == "svm4hog":
        model.svm.svm4hog()
        exit()

    models = []
    train_trans = albumentations.Compose(
        [
            albumentations.Resize(320, 320),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=180, p=0.7),
            albumentations.RandomBrightnessContrast(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.1, rotate_limit=0
            ),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)])
    valid_trans = albumentations.Compose([
            albumentations.Resize(320, 320),
            albumentations.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225],
                max_pixel_value=255.0, always_apply=True
            ),
            ToTensorV2(p=1.0)
        ])
        
    train_valid_data = MyTrainValidDataset(".", "train.csv")

    train_data = MyLeaveDataset(".", "train.csv", mode="train", valid_ratio=0.2, trans=train_trans)
    valid_data = MyLeaveDataset(".", "train.csv", mode="valid", valid_ratio=0.2, trans=valid_trans)
    test_data = MyLeaveDataset(".", "test.csv", mode="test", valid_ratio=0.114514, trans=valid_trans)

    train_iter = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=False)
    valid_iter = DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=False)
    test_iter = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    if args.mode == "train":
        if args.model_type == "resnet34":
            models.append(res_model_34(num_classes, frozen=False, pretrained=args.pretrained))
        elif args.model_type == "resnext50":
            models.append(resnext_model_50(num_classes, frozen=False, pretrained=args.pretrained))
        elif args.model_type == "resnest50":
            models.append(resnest_model_50(num_classes, frozen=False, pretrained=args.pretrained))
        elif args.model_type == "densenet161":
            models.append(densenet_161(num_classes, frozen=False, pretrained=args.pretrained))

        
        criterion = nn.CrossEntropyLoss()

        # criterion = nn.CrossEntropyLoss()

        if len(models) == 1:
            model = models[0]
            if isinstance(model, nn.Module):
                train_deep.train(model, args.num_epoch, train_iter, valid_iter, args.lr, args.wd, criterion, get_device(), ".", mixup=args.mixup, cutmix=args.cutmix, type=args.model_type)
    else:
        print(f"len(test_data){len(test_data)}")
        models.append(resnext_model_50(num_classes))
        models.append(resnest_model_50(num_classes))
        models.append(densenet_161(num_classes))
        models[0].load_state_dict(torch.load("./submitmodels/best_acc0_resnext50_False_True.pth"))
        models[1].load_state_dict(torch.load("./submitmodels/best_acc0_resnest50_False_True.pth"))
        models[2].load_state_dict(torch.load("./submitmodels/best_acc0_densenet161_False_True.pth"))
        predict(models[0], test_iter, get_device(), "./test.csv", "resnext.csv")
        predict(models[1], test_iter, get_device(), "./test.csv", "resnest.csv")
        predict(models[2], test_iter, get_device(), "./test.csv", "densenet.csv")
        sub1, sub2, sub3 = pd.read_csv("./resnext.csv"), pd.read_csv("resnest.csv"), pd.read_csv("./densenet.csv")
        sub = sub1.copy()
        sub.rename(columns={"label":"label_resnext"}, inplace=True)
        sub["label_resnest"] = sub2.copy()["label"]
        sub["label_densenet"] = sub3.copy()["label"]
        sub["label"] = 0
        for rows in range(len(sub)):
            if sub["label_resnext"].iloc[rows] == sub["label_densenet"].iloc[rows]:
                sub["label"].iloc[rows] = sub.copy()["label_resnext"].iloc[rows]
            else:
                sub["label"].iloc[rows] = sub.copy()["label_resnest"].iloc[rows]
        subb = sub.copy()[["image", "label"]]
        subb.to_csv("./submission.csv", index=False)