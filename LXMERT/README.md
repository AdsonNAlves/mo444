Learning Cross-Modality Encoder Representations from Transformers (LXMERT)
===============

```bash
cd LXMERT
mkdir data
ln -s path/to/pvqa/ data/pvqa
cp -r saved/lxmert data/
```

# Running the code

## Pre-trained models

The pre-trained model (870 MB) is available at http://nlp.cs.unc.edu/data/model_LXRT.pth, and can be downloaded with:

```bash
cd LXMERT
cd snap/pretrained 
wget https://nlp.cs.unc.edu/data/model_LXRT.pth -P snap/pretrained
```

Run :
```bash
cd LXMERT
python pvqa.py \
      --train train --valid val  \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --loadLXMERT snap/pretrained/ \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 20 \
      --tqdm --output snap/output
```
```bash
python src/tasks/pvqa.py \
      --test test  --train val --valid " " \
      --load snap/output/BEST \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
      --seed $seed --pvqaimgv $imgv \
      --tqdm --output snap/eval
```


# Pre-training
```bash
python src/pretrain/lxmert_pretrain.py \
      --taskQA_woi --taskVA2 --taskMatched --taskQA \
      --visualLosses obj,attr,feat \
      --wordMaskRate 0.15 --objMaskRate 0.15 \
      --train  pvqa_train --valid pvqa_val \
      --loadLXMERT snap/pretrained/model \
      --llayers 9 --xlayers 5 --rlayers 5 \
      --batchSize 16 --optim bert --lr 1e-4 --epochs 2 \
      --seed $seed --pvqaimgv $imgv \
      --tqdm --output $pre_output
```