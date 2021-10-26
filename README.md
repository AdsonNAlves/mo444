# mo444
Machine learning Course - Team

Download the data and put it into the 'data' file

Download Dataset from : Google Drive Link: https://drive.google.com/file/d/1utnisF_HJ8Yk9Qe9dBe9mxuruuGe7DgW/view?usp=sharing

To train: 
```bash
python finetune_main.py --task pvqa --epoch 10 --start_epoch 0 --lr 0.01 --cos --train train --val val --tfidf --output saved_models\name --batch_size 128
```
