# Enhanced Mixup for Improved Time Series Analysis
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)
![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)

This repository is the official implementation of [Enhanced Mixup for Improved Time Series Analysis](). 

## Installation Instructions

We conducted experiments under
- python 3.8.0

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset
We experiment on 8 benchmark dataset: 
- LTSF: ETTh1, ETTh2, ETTm1, ETTm2
- Classification: Handwriting, Heartbeat, SelfRegulationSCP1, SelfRegulationSCP2

0. Download the zipped data from https://drive.google.com/file/d/1uSbHoCwHt7H6O5YNevSkSNrIiIAlLcXO/view?usp=sharing
1. Extract the zipped data in folder `TS_EMixup`
2. Run a command in the scripts located in the  `TS_EMixup/scripts directory.`


## Training

**To view the complete set of commands used to train models as detailed in the experiment section of our paper, please refer to the files located in the `TS_EMixup/scripts directory.`**

- Example: EMxiup - DLinear - ETTh1
```python
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model DLinear \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --mixup_name enahced_mixup \
  --mixup_rate 1.0 \
  --alpha 0.5 \
  --emixup_vflip_rate 0.15 
```

If you found EMixup useful in your research, please consider starring ⭐ us on GitHub and citing us in your research!

## Acknowledgement
This library is constructed based on the following repo:
- Time Series Library (TSlib): https://github.com/thuml/Time-Series-Library