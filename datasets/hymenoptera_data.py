import os
import zipfile
import requests
import typing
from torchvision import datasets as tv_datasets, transforms
from lib.dataset_data import Datasource
from synth_mind.supervised.data import DatasetGroup

class HymenopteraData(Datasource[tuple[typing.Any, typing.Any]]):
    def __init__(
        self,
        root_path: str,
        max_length: int | None = None,
    ):
        url = 'https://download.pytorch.org/tutorial/hymenoptera_data.zip'

        zip_base_path = f'{root_path}/zip'
        zip_path = f'{zip_base_path}/data.zip'
        zip_dir_path = f'{root_path}'

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

        data_dir = f'{zip_dir_path}/hymenoptera_data'
        image_datasets = {
            x: tv_datasets.ImageFolder(
                os.path.join(data_dir, x),
                data_transforms[x],
            ) for x in ['train', 'val']
        }

        class_names = image_datasets['train'].classes
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        train_dataset = image_datasets['train']
        validation_dataset = image_datasets['val']

        datasets = DatasetGroup(
            train=train_dataset,
            validation=validation_dataset,
            test=None)

        datasets = datasets.limit(max_length)

        super().__init__(datasets=datasets)

        self.data_dir = data_dir
        self.class_names = class_names
        self.dataset_sizes = dataset_sizes
        self.data_transforms = data_transforms
