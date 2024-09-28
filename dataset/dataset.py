import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def train_transform(resize_size=256, crop_size=224):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))

    return transforms.Compose([
        # transforms.Resize(resize_size, interpolation=BICUBIC),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


def test_transform(resize_size=224, crop_size=224):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))
    return transforms.Compose([
        # transforms.Resize(resize_size, interpolation=BICUBIC),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

'''
assume classes across domains are the same.
[0 1 ............................................................................ N - 1]
|---- common classes --||---- src_domain private classes --||---- target private classes --|

|-------------------------------------------------|
|                DATASET PARTITION                |
|-------------------------------------------------|
|DATASET    |  class split(com/sou_pri/tar_pri)   |
|-------------------------------------------------|
|DATASET    |    PDA    |    OSDA    |   OPDA    |
|-------------------------------------------------|
|OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
|-------------------------------------------------|
|VisDA-2017 |           |   6/0/6    |   6/3/3    |
|-------------------------------------------------|  
|DomainNet  |           |            | 150/50/145 |
|-------------------------------------------------|
'''

class SFUniDADataset(Dataset):

    def __init__(self, args, img_dir, img_list, domain_type, preload_flg=True) -> None:
        super(SFUniDADataset, self).__init__()

        self.domain_type = domain_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg

        self.num_common_classes = args.num_common_classes
        self.num_source_private_classes = args.num_source_private_classes
        self.num_target_private_classes = args.num_target_private_classes

        self.common_classes = [i for i in range(args.num_common_classes)]
        self.source_private_classes = [i + args.num_common_classes for i in range(args.num_source_private_classes)]

        self.target_private_classes = [i + args.num_common_classes + args.num_source_private_classes
                                       for i in range(args.num_target_private_classes)]

        self.source_classes = self.common_classes + self.source_private_classes
        self.target_classes = self.common_classes + self.target_private_classes

        self.img_dir = img_dir
        self.imgs = [item.strip().split() for item in img_list]

        # Filtering the img_list
        if self.domain_type == "src_domain":
            # self.img_dir = args.source_dir
            self.imgs = [item for item in self.imgs if int(item[1]) in self.source_classes]
        else:
            # self.img_dir = args.target_dir
            self.imgs = [item for item in self.imgs if int(item[1]) in self.target_classes]

        if args.clip_encoder in ['RN50', 'ViT-B/16']:
            self.train_transform = train_transform()
            self.test_transform = test_transform()
            self.train_size = 256
            self.test_size = 224
        elif args.clip_encoder == 'RN50x16':
            self.train_transform = test_transform(384,384)
            self.test_transform = test_transform(384,384)
            self.train_size = 384
            self.test_size = 384

        self.preload()

    def preload(self):
        if self.preload_flg:
            resize_trans = transforms.Resize(self.train_size, interpolation=BICUBIC)
            print("Dataset Pre-Loading Started ....")
            self.train_imgs = [resize_trans(Image.open(os.path.join(self.img_dir, item[0])).copy().convert("RGB"))
                               for item in tqdm(self.imgs, ncols=60)]
            if self.train_size != self.test_size:
                resize_trans = transforms.Resize(self.test_size, interpolation=BICUBIC)
                self.test_imgs = [resize_trans(Image.open(os.path.join(self.img_dir, item[0])).copy().convert("RGB"))
                                  for item in tqdm(self.imgs, ncols=60)]
            else:
                self.test_imgs = None
            print("Dataset Pre-Loading Done!")
        else:
            pass

    def load_img(self, idx):
        img_f, img_label = self.imgs[idx]
        if "officehome" in self.dataset and self.preload_flg:
            train_img = self.train_imgs[idx]
            if self.test_imgs is not None:
                test_img = self.test_imgs[idx]
            else:
                test_img = self.train_imgs[idx]
        elif "visda" in self.dataset and self.preload_flg:
            train_img = self.train_imgs[idx]
            if self.test_imgs is not None:
                test_img = self.test_imgs[idx]
            else:
                test_img = self.train_imgs[idx]
        elif "domainnet" in self.dataset and self.preload_flg:
            train_img = self.train_imgs[idx]
            if self.test_imgs is not None:
                test_img = self.test_imgs[idx]
            else:
                test_img = self.train_imgs[idx]
        else:
            resize_trans = transforms.Resize(self.train_size, interpolation=BICUBIC)
            train_img = resize_trans(Image.open(os.path.join(self.img_dir, img_f)).copy().convert("RGB"))

            resize_trans = transforms.Resize(self.test_size, interpolation=BICUBIC)
            test_img = resize_trans(Image.open(os.path.join(self.img_dir, img_f)).copy().convert("RGB"))
        return train_img, test_img, img_label

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):

        train_img, test_img, label = self.load_img(idx)

        if self.domain_type == "source":
            label = int(label)
        else:
            label = int(label) if int(label) in self.source_classes else len(self.source_classes)

        train_img = self.train_transform(train_img)
        test_img = self.test_transform(test_img)

        return train_img, test_img, label, idx