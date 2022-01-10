# Keyword Spotting with BCResnet and Arcface Loss 
===============================================================================
- [Broadcasted Residual Learning for Efficient Keyword Spotting](https://arxiv.org/abs/2106.04140)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
## Prepare Data

```bash
python3 -m prepare_data.GSC
```
## Training 
```bash
CUDA_VISIBLE_DEVICES=0 python3 -m train.train --metric=arcface --num_epoch=2 --no_evaluate
```
## Config
- To change some parameters, go config/config.py 












