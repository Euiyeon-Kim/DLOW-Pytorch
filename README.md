# DLOW-Pytorch
Reproduce DLOW: Domain Flow for Adaptation and Generalization

## Code structure
	├─ dataset
	│  ├─ train
	│  │  ├─ Source
	│  │  └─ Target
    │  ├─ val
	│  │  ├─ Source
	│  │  └─ Target
	│  └─ test
	│     ├─ Source
	│     └─ Target
	├─ data
	│		DataLoader.py : transformer 포함 / dataset의 파일 형식 지정 (.jpg or .png)
	│		preprocess.py
	├─ model
	│		checkpoint
	│		BaseNetwork.py
	│		Segmentation.py
	│		Interpolation.py
	│		DLOW.py
	├─ utils
	│		Buffer.py : Discriminator를 학습시킬 때 사용할 버퍼
	│		Logger.py : Terminal + Visdom Logging 
	├─ log
    │		train.log
	│		evaluate.log
	├─ train.py
	└─ evaluate.py

