from torchvision import transforms, datasets
import os
import torch
from PIL import Image

def ImageNetData(args):
# data_transform, pay attention that the input of Normalize() is Tensor and the input of RandomResizedCrop() or RandomHorizontalFlip() is PIL Image
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(args.data_dir, 'ILSVRC2012_img_train'), data_transforms['train'])
    image_datasets['val'] = ImageNetValDataSet(os.path.join(args.data_dir, 'ILSVRC2012_img_val'),
                                               os.path.join(args.data_dir, 'ILSVRC2012_devkit_t12', 'data','ILSVRC2012_validation_ground_truth.txt'),
                                               data_transforms['val'])

    # wrap your data and label into Tensor
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 shuffle=True,
                                                 num_workers=args.num_workers) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloders, dataset_sizes

class ImageNetValDataSet(torch.utils.data.Dataset):
    def __init__(self, img_path, img_label, data_transforms):
        self.data_transforms = data_transforms
        img_names = os.listdir(img_path)
        self.img_path = [os.path.join(img_path, img_name) for img_name in img_names]
        with open(img_label,"r") as input_file:
            lines = input_file.readlines()
            self.img_label = [(int(line)-1) for line in lines]

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = Image.open(self.img_path[item]).convert('RGB')
        label = self.img_label[item]
        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img)
            except:
                print("Cannot transform image: {}".format(self.img_path[item]))
        return img, label
