# FER_POSTER

### Preparation
- create conda environment (we provide requirements.txt)

- Data Preparation

  Download [RAF-DB](http://www.whdeng.cn/RAF/model1.html#dataset) dataset, and make sure it have a structure like following:
 
	```
	- data/raf-basic/
		 EmoLabel/
		     list_patition_label.txt
		 Image/aligned/
		     train_00001_aligned.jpg
		     test_0001_aligned.jpg
		     ...
	```

- Pretrained model weights
Dowonload pretrain weights (Image backbone and Landmark backbone) from [here](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing). Put entire `pretrain` folder under `models` folder.

	```
	- models/pretrain/
		 ir50.pth
		 mobilefacenet_model_best.pth.tar
		     ...
	```

### Testing

Our best model can be download from [here](https://drive.google.com/drive/folders/1jeCPTGjBL8YgKKB9YrI9TYZywme8gymv?usp=sharing), put under `checkpoint ` folder. You can evaluate our model on RAD-DB dataset by running: 

```
python test.py --checkpoint checkpoint/rafdb_best.pth -p
```

### Training
Train on RAF-DB dataset:
```
python train.py --gpu 0,1,2,3 --batch_size 400
```
You may adjust batch_size based on your # of GPUs. Usually bigger batch size can get higher performance. We provide the log in  `log` folder. You may run several times to get the best results. 


## License

Our research code is released under the MIT license. See [LICENSE](LICENSE) for details. 



## Acknowledgments

Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[JiaweiShiCV/Amend-Representation-Module](https://github.com/JiaweiShiCV/Amend-Representation-Module) 


