# nuaa_data_mining_course_design
NUAA数据挖掘课设

树叶分类：https://www.kaggle.com/competitions/classify-leaves

No copies, each python file was written by myself (with chat-gpt maybe:))

## Requirements

This repo was tested on Python 3.8+ and PyTorch 2.0. The main requirements are:

- tqdm
- scikit-learn
- pytorch >= 2.0.0
- albumentations>=1.3.1

To get the environment settled, run:

```
pip install -r requirements.txt
```

## Usage

For traditional models:

```
python main.py --model_type=$model_name
```

model_name can be: dummy、knn4sift、knn4hog、svm4sift、svm4hog

For DL models:

```
python main.py --model_type=$model_name --pretrained
python main.py --model_type=$model_name --pretrained --cutmix #using cutmix
python main.py --model_type=$model_name --pretrained --mixup #using mixup
```

model_name can be: resnet34、resnext50、resnest50、densenet161

For test:
```
python main.py --mode=test
```
to generate `submission.csv`.

For detail, see `main.py`.

## 三种方法

老师要求要用三种不同的方法来解决问题，我觉得k折交叉验证+特征提取+聚类/降维+传统/深度学习分类器+ensemble应该够了吧。。。

另外，希望这是我最后一次做ai相关的项目，感觉做的纯一拖。哥们还是更喜欢做sys。
