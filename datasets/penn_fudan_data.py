import os
import torch
import zipfile
import requests
import typing
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torch.utils.data import Subset
from torchvision.transforms import v2 as T
from synth_mind.supervised.data import DatasetGroup
from lib.dataset_data import Datasource

def get_transform(train: bool):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class PennFudanDataset(torch.utils.data.Dataset[tuple[typing.Any, typing.Any]]):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = read_image(img_path)
        mask = read_image(mask_path)
        # instances are encoded as different colors
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)

        # split the color-encoded mask into a set
        # of binary masks
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        # get bounding box coordinates for each mask
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)

        image_id = idx
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Wrap sample and targets into torchvision tv_tensors:
        img = tv_tensors.Image(img)

        target = {}
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)) # type: ignore
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class PennFudanData(Datasource[tuple[typing.Any, typing.Any]]):
    def __init__(
            self,
            root_path: str,
            max_length: int | None = None):

        url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'

        zip_base_path = f'{root_path}/zip'
        zip_path = f'{zip_base_path}/data.zip'
        zip_dir_path = f'{root_path}/data'

        # verify if dir_path is empty
        if not os.path.exists(zip_dir_path):
            if not os.path.exists(zip_path):
                os.makedirs(os.path.dirname(zip_path), exist_ok=True)

                for file in os.listdir(zip_base_path):
                    file_path_tmp = os.path.join(zip_base_path, file)
                    if os.path.isfile(file_path_tmp):
                        os.unlink(file_path_tmp)

                response = requests.get(url)

                # download zip to zip_path
                with open(zip_path, 'wb') as f:
                    f.write(response.content)

            # extract to zip_base_path
            # (the directory extracted will be zip_dir_path)
            os.makedirs(zip_base_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_dir_path)

        train_dataset = PennFudanDataset(f'{zip_dir_path}/PennFudanPed', get_transform(train=True))
        test_dataset = PennFudanDataset(f'{zip_dir_path}/PennFudanPed', get_transform(train=False))

        # split the dataset in train, validation and test set
        indices = torch.randperm(len(train_dataset)).tolist()
        train_dataset = Subset(train_dataset, indices[:-50])
        validation_dataset = Subset(test_dataset, indices[-50:-25])
        test_dataset = Subset(test_dataset, indices[-25:])

        datasets = DatasetGroup(
            train=train_dataset,
            validation=validation_dataset,
            test=test_dataset)

        datasets = datasets.limit(max_length)

        super().__init__(datasets=datasets)

        self.get_transform = get_transform
        self.main_data_dir = f'{zip_dir_path}/PennFudanPed'
