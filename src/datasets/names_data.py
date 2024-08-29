import os
import zipfile
from typing import Tuple
import requests
from auto_mind.supervised.data import ItemsDataset, SplitData
from src.lib.dataset_data import LabeledDatasource
from src.lib import data_utils

class NamesData(LabeledDatasource[str, int, str]):
    def __init__(self, root_path: str, split_data: SplitData, random_seed: int | None) -> None:
        url = 'https://download.pytorch.org/tutorial/data.zip'

        zip_base_path = f'{root_path}/zips/names'
        zip_path = f'{zip_base_path}/data.zip'
        zip_dir_path = f'{zip_base_path}/data'
        dir_path = f'{root_path}/names'

        # verify if dir_path is empty
        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            if not os.path.exists(zip_path):
                os.makedirs(os.path.dirname(zip_path), exist_ok=True)

                for file in os.listdir(zip_base_path):
                    file_path = os.path.join(zip_base_path, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)

                response = requests.get(url, timeout=1000)

                # download zip to zip_path
                with open(zip_path, 'wb') as f:
                    f.write(response.content)

            # extract to zip_base_path
            # (the directory extracted will be zip_dir_path)
            os.makedirs(zip_base_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_base_path)

            # delete the main directory dir_path
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                os.rmdir(dir_path)

            # move zip_dir_path/names to dir_path
            os.rename(f'{zip_dir_path}/names', dir_path)

        files_glob_path = root_path + '/names/*.txt'

        # Build the category_lines dictionary, a list of lines per category
        items: list[Tuple[str, int]] = []
        all_labels: list[str] = []

        for filename in data_utils.find_files(files_glob_path):
            category = os.path.splitext(os.path.basename(filename))[0]
            cat_idx = len(all_labels)
            all_labels.append(category)
            lines = data_utils.read_lines(filename)
            items += [(line, cat_idx) for line in lines]

        if not all_labels:
            raise Exception('could not load categories')

        dataset = ItemsDataset(items=items)
        datasets = split_data.split(
            dataset=dataset,
            shuffle=bool(split_data.val_percent or split_data.test_percent),
            random_seed=random_seed)

        super().__init__(datasets=datasets, all_labels=all_labels)
