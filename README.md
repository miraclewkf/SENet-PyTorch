**This is the PyTorch implement of SENet(Squeeze-and-Excitation Networks)**[https://arxiv.org/pdf/1709.01507.pdf]

### Project

```
├── train.py
├── SENet.py
├── read_ImageNetData.py
├── ImageData
		├── ILSVRC2012_img_train
					├── n01440764
					├──    ...
					├── n15075141
		├── ILSVRC2012_img_val
		├── ILSVRC2012_dev_kit_t12
					├── data
						├── ILSVRC2012_validation_ground_truth.txt
├── output
```

**You should download ImageNet dataset and put them as in `ImageData` folder as above.**


### Usage

* Train from scratch using ImageNet dataset:

```
train train.py --batch-size 64
```