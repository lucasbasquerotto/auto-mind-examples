import os
import zipfile
import requests
from auto_mind.supervised.data import ItemsDataset, SplitData
from src.lib.dataset_data import Datasource
from src.lib import data_utils

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def _filter_pair(
    pair: tuple[str, str],
    max_length: int | None,
    filter_sentences: bool,
) -> bool:
    input = data_utils.normalize_string(pair[0])
    target = data_utils.normalize_string(pair[1])

    valid_prefix = (
        (not filter_sentences)
        or
        (input.startswith(eng_prefixes))
        or
        (target.startswith(eng_prefixes))
    )

    if max_length is None:
        return valid_prefix

    return (
        valid_prefix and
        (len(input.split(' ')) < max_length) and
        (len(target.split(' ')) < max_length))


def _filter_pairs(
    pairs: list[tuple[str, str]],
    max_length: int | None,
    filter_sentences: bool,
) -> list[tuple[str, str]]:
    return [
        (pair[0], pair[1])
        for pair in pairs
        if _filter_pair(pair=pair, max_length=max_length, filter_sentences=filter_sentences)]

class TranslationData(Datasource[tuple[str, str]]):
    def __init__(
        self,
        root_path: str,
        lang1: str,
        lang2: str,
        split_data: SplitData,
        max_length: int | None = None,
        reverse: bool = False,
        normalize: bool = False,
        autoencoder: bool = False,
        filter_sentences: bool = True,
        random_seed: int | None = None,
    ) -> None:
        url = 'https://download.pytorch.org/tutorial/data.zip'

        file_name = f'{lang1}-{lang2}.txt'
        zip_base_path = f'{root_path}/zips/names'
        zip_path = f'{zip_base_path}/data.zip'
        zip_dir_path = f'{zip_base_path}/data'
        file_path = f'{root_path}/seq2seq/{file_name}'

        # verify if dir_path is empty
        if not os.path.exists(file_path):
            if not os.path.exists(zip_path):
                os.makedirs(os.path.dirname(zip_path), exist_ok=True)

                for file in os.listdir(zip_base_path):
                    file_path_tmp = os.path.join(zip_base_path, file)
                    if os.path.isfile(file_path_tmp):
                        os.unlink(file_path_tmp)

                response = requests.get(url, timeout=1000)

                # download zip to zip_path
                with open(zip_path, 'wb') as f:
                    f.write(response.content)

            # extract to zip_base_path
            # (the directory extracted will be zip_dir_path)
            os.makedirs(zip_base_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(zip_base_path)

            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok=True)

            # move zip_dir_path/names to dir_path
            os.rename(f'{zip_dir_path}/{file_name}', file_path)

        # Read the file and split into lines
        lines = open(
            f'{file_path}',
            encoding='utf-8').read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs: list[tuple[str, str]] = [
            ((item[1], item[0]) if reverse else (item[0], item[1]))
            for item in (l.split('\t') for l in lines)]

        pairs = _filter_pairs(
            pairs=pairs,
            max_length=max_length,
            filter_sentences=filter_sentences)

        if normalize:
            pairs = [
                (data_utils.normalize_string(input), data_utils.normalize_string(output))
                for input, output in pairs
            ]

        # if it will be used by an autoencoder, the input and output will be the same
        # lang1 will be used if reverse is False, otherwise it will be lang2
        if autoencoder:
            pairs = [(p[0], p[0]) for p in pairs]

        dataset = ItemsDataset(items=pairs)
        datasets = split_data.split(dataset=dataset, random_seed=random_seed)

        super().__init__(datasets=datasets)
