# image_retrieval

Small framework to conduct experiment for image retrieval with metric learning, softmax and SLL approach


## How to use it ?


supervised learning 

```bash
python image_retrieval/script/train.py 10 
```

SSL

```bash
python image_retrieval/script/train.py 900 --num-workers 16 --batch-size 512 --patience 100 --lr 0.06 --aug "SSLAugmentation2" --module "SimSiamModule" --backbone "ResNet18"
```
