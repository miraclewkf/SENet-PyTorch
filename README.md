**This is the PyTorch implement of SENet**([Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf))

### Project

```
├── train.py # train script
├── SENet.py # network of SENet
├── read_ImageNetData.py # ImageNet dataset read script
├── ImageData # train and validation data
	├── ILSVRC2012_img_train
		├── n01440764
		├──    ...
		├── n15075141
	├── ILSVRC2012_img_val
	├── ILSVRC2012_dev_kit_t12
		├── data
			├── ILSVRC2012_validation_ground_truth.txt
```

**You should download ImageNet dataset and put them in `ImageData` folder as above.**


### Usage

* Train from scratch using ImageNet dataset:

```
python train.py --batch-size 64 --gpus 0,1 --lr 0.1
```