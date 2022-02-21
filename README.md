# TACnet: Video Anomaly Detection by Temporal Attention Clustering Network
This repo is the offical implementation of this [paper](https://drive.google.com/uc?export=download&id=13ZapwcvD6-HNEcJmo9kbahvLC9ift9KP).
## Dataset
### UCF Crime
This dataset is proposed by [Sultani et al.](https://www.crcv.ucf.edu/projects/real-world/) To run this repo, simply download the dataset through the [link](https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0) and unzip to desired location.

### ShanghaiTech
The download [link](https://svip-lab.github.io/dataset/campus_dataset.html) of this dataset is proposed by [Liu et al.](https://svip-lab.github.io/dataset/campus_dataset.html) In addition, because the dataset is proposed for unsupervised learning task, we used the new split by [GCN-Anomaly](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection). The split [file](https://github.com/jx-zhong-for-academic-purpose/GCN-Anomaly-Detection/tree/master/ShanghaiTech_new_split) should be put in the SHtech dataset folder.

## How to use
- update the dataset folder path in **src/config.py**  
- download the pretrained weights (Sports1M) of C3D from [here](http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle) and put into **models/**  

### Preprocessing
- convert video to frames
```
python preprocess/videotoframe_UCF.py [--reset] #for UCFCrime
python preprocess/videotoframe_SH.py [--reset] #for ShanghaiTech
```
[--reset]: clean all genearted folders

- slice the frames into clips
In our setting, all videos are sliced into 32 segments and each segment are sampled into 16 frames.
```
python preprocess/clip_UCF.py #for UCFCrime
python preprocess/clip_SH.py # for ShanghaiTech
```

### Training
```
python main.py [--dataset] [--savelog] [--name] [--note] [--model_path] [--attention_type] [--gpus] [--epoch] [--lr] [--batch_size]
```
[--dataset] (**required** str): UCF or SH  
[--savelog] (store_true): The log file would be saved as **{dataset_folder}/log/{name}.log** and the trained model weights would be saved in **{dataset_folder}/log/{name}/**. The default name is the datetime.  
[--name] (str): set the name of logfile  
[--note] (str): available to add some notes in logfile.  
[--model_path] (Path): Trained model weights' path if desired to resume training.  
[--attention type] (str): gated or normal. The type of attention mechanism in TACnet.   
[--gpus] (str): choose the training GPU. This doesn't support for multi GPUs training.  

### Visualize the results
```
python predict.py [--dataset] [--model_path] [--load_pretrain] [--gpus] [--p_graph] [--c_grpah] [--attn_graph]
```
[--dataset] (**required** str): UCF or SH  
[--model_path] (Path): Trained model weights' path  
[--load_pretrain] (store_true): use the pretrained C3D weights or fine-tuned C3D weights.  
[--p_graph] (store_true): output the performance graph of each testcase to **{dataset_folder}/image/{name}/performance/** 
<img src="https://i.imgur.com/3lQWMtM.png" height="400">

[--c_graph] (store_true): output the TSNE visualized cluster graph of each testcase to **{dataset_folder}/image/{name}/cluster/**  
<img src="https://i.imgur.com/Mepbbqy.png" height="400">

[--attn_graph] (store_true): output the visualization of attention weights of each testcase to **{dataset_folder}/image/{name}/attn/**  
![](https://i.imgur.com/sGmzKmE.png)

