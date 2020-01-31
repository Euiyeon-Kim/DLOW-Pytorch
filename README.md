# DLOW-Pytorch
Reproduce [DLOW: Domain Flow for Adaptation and Generalization](https://pdfs.semanticscholar.org/cbe1/a8b4712654f382192dc1ccaf00ddfc12f57b.pdf).



## Code structure
	├─ dataset
	│  ├─ Source
	│  │  ├─ img
	│  │  └─ label
	│  ├─ Source_list
	│  │  └─ train.txt
	│  │
	│  ├─ Target
	│  │  ├─ train
	│	 │  │  ├─ img
	│  │  │  └─ label
	│  │  ├─ test
	│	 │  │  ├─ img
	│  │  │  └─ label
	│  │  └─ val
	│	 │     ├─ img
	│  │     └─ label
	│  └─ Target_list
	│     ├─ info.json
	│     ├─ label.txt
	│     ├─ train.txt
	│     └─ val.txt
	│
	├─ data
	│		DataLoader.py
	│		preprocess_Cityscapes.py 	: Resizing
	│		preprocess_GTA5.py 			: Split train, test, valid dataset and resizing
	│
	├─ model
	│		checkpoint
	│		Modules.py
	│		BaseNetwork.py
	│		InterpolationGAN.py
	│
	├─ utils
	│		Logger.py 					: Terminal + tensorboard Logging 
	│		utils.py
	│
	├─ train.py							: Train InterpolationGAN
	│
	└─ infer.py 						: Make actual DLOW dataset using InterpolationGAN
	
	
	


## How to execute

### 1. Prepare dataset

   dataset 폴더에 아래와 같은 구조로 Cityscapes dataset( leftImg8bit )과 GTA5 dataset 준비

   Cityscapes의 img, label 디렉토리 내부에는 도시 이름으로 된 폴더들 존재

   + [Cityscapes](https://www.cityscapes-dataset.com/)
   + [GTA5](https://download.visinf.tu-darmstadt.de/data/from_games/)

   ```
   └─ dataset
      ├─ Cityscapes
      │  ├─ test
      │  │  ├─ img 		: From leftImg8bit_trainvaltest.zip
      │  │  └─ label		: From gtFine_trainvaltest.zip
      │  ├─ train
      │  │  ├─ img
      │  │  └─ label
      │  └─ val
      │     ├─ img
      │     └─ label   
      └─ GTA5
         ├─ img
         └─ label
   ```



### 2. Data preprocessing

   데이터 셋을 train, test, validation set으로 나누고 resizing

   ~~~python
   cd data
   python3 preprocess_GTA5.py
   python3 preprocess_Cityscapes.py
   ~~~

   정상 동작시 아래와 같은 구조의 Source와 Target폴더가 생성됨

   ```
   └─ dataset
      ├─ Source
      │  ├─ test
      │  │  ├─ img 						
      │  │  └─ label										
      │  ├─ train
      │  │  ├─ img
      │  │  └─ label
      │  └─ val
      │     ├─ img
      │     └─ label   
      └─ Target
         ├─ test
         │  ├─ img 						
         │  └─ label										
         ├─ train
         │  ├─ img
         │  └─ label
         └─ val
            ├─ img
            └─ label   
   ```



### 3. Run train.py

   DLOW를 생성하는 InterpolationGAN 학습

   실시간 학습 현황을 보고 싶다면 tensorboard 실행

   ~~~python
   python3 train.py
   tensorboard --logdir=./runs
   ~~~



### 4. Run infer.py

   3에서 학습한 InterpolationGAN을 활용해 데이터셋 생성

   ~~~python
   python3 infer.py
   ~~~



### 5. Performance measurement

   4에서 생성한 DLOW 데이터셋이 일반 데이터셋을 사용하는 것 보다 얼마나 좋은지 성능 비교하기

   코드 및 설명 참조 : https://github.com/wasidennis/AdaptSegNet

   ~~~python
   python3 ./AdaptSegNet/before.py     # 기존 데이터셋을 이용한 학습
   python3 ./AdaptSegNet/after.py      # 새로 생성한 데이터셋을 이용한 학습
   ~~~
