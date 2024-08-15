import math
import random
import string
import typing
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Callable, Generic, TypeVar
from torch.nn.utils.rnn import pad_sequence
from synth_mind.supervised.data import DatasetGroup, IterDataset, SplitData
from lib import data_utils

T = TypeVar("T")
I = TypeVar("I")
O = TypeVar("O")

####################################################
#################### Vocabulary ####################
####################################################


class Vocab(Generic[I]):
    def __init__(self, name: str | None, unk: I | None, pad: I | None, bos: I | None, eos: I | None):
        self.name = name
        self.n_tokens = 0
        self.specials: list[I] = []
        self.index2token: dict[int, I] = dict()
        self.token2index = {}
        self.token2count = {}

        self.unk_idx: int | None = None
        self.pad_idx: int | None = None
        self.bos_idx: int | None = None
        self.eos_idx: int | None = None

        self.unk_token: I | None = None
        self.pad_token: I | None = None
        self.bos_token: I | None = None
        self.eos_token: I | None = None

        self.define_specials(unk=unk, pad=pad, bos=bos, eos=eos)

    def define_specials(self, unk: I | None, pad: I | None, bos: I | None, eos: I | None):
        unk_idx: int | None = None
        pad_idx: int | None = None
        bos_idx: int | None = None
        eos_idx: int | None = None

        self.unk_token = unk
        self.pad_token = pad
        self.bos_token = bos
        self.eos_token = eos

        n_tokens = self.n_tokens
        index2token = self.index2token
        specials: list[I] = []

        if unk is not None:
            if self.unk_token is not None and self.unk_idx is not None:
                unk_idx = self.unk_idx
            else:
                unk_idx = n_tokens
                n_tokens += 1
                self.unk_idx = unk_idx

            index2token[unk_idx] = unk
            specials.append(unk)

        if pad is not None:
            if self.pad_token is not None and self.pad_idx is not None:
                pad_idx = self.pad_idx
            else:
                pad_idx = n_tokens
                n_tokens += 1
                self.pad_idx = pad_idx

            index2token[pad_idx] = pad
            specials.append(pad)

        if bos is not None:
            if self.bos_token is not None and self.bos_idx is not None:
                bos_idx = self.bos_idx
            else:
                bos_idx = n_tokens
                n_tokens += 1
                self.bos_idx = bos_idx

            index2token[bos_idx] = bos
            specials.append(bos)

        if eos is not None:
            if self.eos_token is not None and self.eos_idx is not None:
                eos_idx = self.eos_idx
            else:
                eos_idx = n_tokens
                n_tokens += 1
                self.eos_idx = eos_idx

            index2token[eos_idx] = eos
            specials.append(eos)

        self.n_tokens = n_tokens
        self.specials = specials

    def add_token(self, token: I):
        if token not in self.token2index:
            self.token2index[token] = self.n_tokens
            self.token2count[token] = 1
            self.index2token[self.n_tokens] = token
            self.n_tokens += 1
        else:
            self.token2count[token] += 1

    def token2idx(self, token: I) -> int | None:
        idx = self.token2index.get(token)
        idx = idx if idx is not None else self.unk_idx
        return idx

    def tokens2idxs(self, tokens: list[I]) -> list[int]:
        idxs = [self.token2idx(token) for token in tokens] if tokens else []
        return [idx for idx in idxs if idx is not None]

    def idxs2tokens(
            self,
            idxs: list[int],
            include_unk=False,
            include_pad=False,
            include_bos=False,
            include_eos=False,
            continue_after_eos=False) -> list[I]:
        tokens: list[I] = []

        for idx in idxs:
            if self.eos_idx is not None and idx == self.eos_idx:
                if include_eos and self.eos_token:
                    tokens.append(self.eos_token)

                if not continue_after_eos:
                    break
            else:
                if idx == self.unk_idx:
                    if include_unk and self.unk_token:
                        tokens.append(self.unk_token)
                elif idx == self.pad_idx:
                    if include_pad and self.pad_token:
                        tokens.append(self.pad_token)
                elif idx == self.bos_idx:
                    if include_bos and self.bos_token:
                        tokens.append(self.bos_token)
                else:
                    tokens.append(self.index2token[idx])

        return tokens


class BareLang(Vocab[str]):
    def __init__(
            self,
            name: str | None = None,
            unk: str | None = None,
            pad: str | None = None,
            bos: str | None = None,
            eos: str | None = None):
        super().__init__(
            name=name,
            unk=unk,
            pad=pad,
            bos=bos,
            eos=eos)

    def define_all_specials(self):
        self.define_specials(
            unk="<unk>",
            pad="<pad>",
            bos="<bos>",
            eos="<eos>")

    def add_sentence(self, sentence: str):
        for token in sentence.split(' '):
            self.add_token(token)


class Lang(BareLang):
    def __init__(self, name: str | None):
        super().__init__(name=name)
        self.define_all_specials()

    def add_sentence(self, sentence: str):
        for token in sentence.split(' '):
            self.add_token(token)


class LettersVocab(Vocab[str]):
    def __init__(self, all_letters: str, has_bos: bool | None = True, has_eos: bool | None = True):
        super().__init__(
            name=None,
            unk='?',
            pad=' ',
            bos='<' if has_bos != False else None,
            eos='>' if has_eos != False else None)

        for char in all_letters:
            self.add_token(char)

        self.all_letters = all_letters


class AsciiLettersData(LettersVocab):
    def __init__(self, has_bos: bool | None = None, has_eos: bool | None = None):
        super().__init__(
            all_letters=string.ascii_letters + " .,;'-",
            has_bos=has_bos,
            has_eos=has_eos)


####################################################
################### Data Handler ###################
####################################################

class DataHandler(Generic[I]):
    def __init__(self, to_tensor: Callable[[I], Tensor], from_tensor: Callable[[Tensor], I]):
        self.to_tensor = to_tensor
        self.from_tensor = from_tensor


class FullDataHandler(DataHandler[I], Generic[I]):
    def __init__(
            self,
            to_tensor: Callable[[I], Tensor],
            from_tensor: Callable[[Tensor], I],
            to_input_tensor: Callable[[I], Tensor],
            from_input_tensor: Callable[[Tensor], I],
            to_target_tensor: Callable[[I], Tensor],
            from_target_tensor: Callable[[Tensor], I]):
        self.to_tensor = to_tensor
        self.from_tensor = from_tensor
        self.to_input_tensor = to_input_tensor
        self.from_input_tensor = from_input_tensor
        self.to_target_tensor = to_target_tensor
        self.from_target_tensor = from_target_tensor


class SizedDataHandler(FullDataHandler[I], Generic[I]):
    def __init__(
            self,
            size: int,
            to_tensor: Callable[[I], Tensor],
            from_tensor: Callable[[Tensor], I],
            to_input_tensor: Callable[[I], Tensor],
            from_input_tensor: Callable[[Tensor], I],
            to_target_tensor: Callable[[I], Tensor],
            from_target_tensor: Callable[[Tensor], I]):

        super().__init__(
            to_tensor=to_tensor,
            from_tensor=from_tensor,
            to_input_tensor=to_input_tensor,
            from_input_tensor=from_input_tensor,
            to_target_tensor=to_target_tensor,
            from_target_tensor=from_target_tensor)

        self.size = size

L = TypeVar("L")

class LabelDataHandler(typing.Generic[L], SizedDataHandler[int]):
    def __init__(self, all_labels: list[L], device: torch.device | None = None):

        def to_tensor(label_idx: int):
            return torch.tensor([label_idx], dtype=torch.long, device=device)

        def from_tensor(tensor: torch.Tensor) -> int:
            return int(tensor[0].item())

        def to_input_tensor(label_idx: int):
            n_labels = len(all_labels)
            tensor = torch.zeros(1, n_labels, device=device)
            tensor[0][label_idx] = 1
            return tensor

        def from_input_tensor(tensor: torch.Tensor):
            _, topi = tensor.topk(1)
            label_idx = int(topi[0].item())
            return label_idx

        def to_target_tensor(label_idx: int):
            return torch.tensor([label_idx], dtype=torch.long, device=device).view(1, -1)

        def from_target_tensor(tensor: torch.Tensor):
            return int(tensor[0][0].item())

        super().__init__(
            to_tensor=to_tensor,
            from_tensor=from_tensor,
            to_input_tensor=to_input_tensor,
            from_input_tensor=from_input_tensor,
            to_target_tensor=to_target_tensor,
            from_target_tensor=from_target_tensor,
            size=len(all_labels))

        self.all_labels = all_labels

class VocabDataHandler(SizedDataHandler[list[I]], Generic[I]):
    def __init__(self, vocab: Vocab[I], flat_input=False, device: torch.device | None = None):

        def to_tensor(tokens: list[I]):
            indexes = vocab.tokens2idxs(tokens=tokens)

            if vocab.eos_idx:
                indexes.append(vocab.eos_idx)

            return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

        def from_tensor(tensor: torch.Tensor) -> list[I]:
            return vocab.idxs2tokens(
                idxs=[int(idx.item()) for idx in tensor.squeeze()])

        def to_input_tensor(tokens: list[I]):
            has_bos = vocab.bos_idx is not None
            size = len(tokens) if not has_bos else len(tokens) + 1
            tensor = torch.zeros(size, 1, vocab.n_tokens, device=device)

            if has_bos:
                tensor[0][0][vocab.bos_idx] = 1

            for li, token in enumerate(tokens):
                idx = li if not has_bos else li + 1
                tensor[idx][0][vocab.token2index[token]] = 1

            if flat_input:
                _, idxs_tensor = tensor.squeeze(1).topk(1)
                tensor = idxs_tensor.squeeze(1)

            return tensor

        def from_input_tensor(tensor: torch.Tensor) -> list[I]:
            result: list[I] = []

            for i in range(tensor.size()[0]):
                _, topi = tensor[i].topk(1)
                token_idx = int(topi[0].item())

                if vocab.eos_idx is not None and token_idx == vocab.eos_idx:
                    break

                token = vocab.index2token[token_idx]

                if token is None:
                    if vocab.unk_token is not None:
                        result.append(vocab.unk_token)
                else:
                    result.append(token)

            return result

        def to_target_tensor(tokens: list[I]):
            has_bos = vocab.bos_idx is not None
            letter_indexes = [
                vocab.token2index[tokens[li]]
                for li in range(0 if has_bos else 1, len(tokens))]

            if vocab.eos_idx is not None:
                letter_indexes.append(vocab.eos_idx)

            return torch.LongTensor(letter_indexes, device=device)

        def from_target_tensor(tensor: torch.Tensor) -> list[I]:
            result: list[I] = []

            for i in range(tensor.size()[0]):
                token_idx = int(tensor[i].item())

                if vocab.eos_idx and token_idx == vocab.eos_idx:
                    break

                token = vocab.index2token[token_idx]

                if token is None:
                    if vocab.unk_token is not None:
                        result.append(vocab.unk_token)
                else:
                    result.append(token)

            return result

        super().__init__(
            size=vocab.n_tokens,
            to_tensor=to_tensor,
            from_tensor=from_tensor,
            to_input_tensor=to_input_tensor,
            from_input_tensor=from_input_tensor,
            to_target_tensor=to_target_tensor,
            from_target_tensor=from_target_tensor)

        self.vocab = vocab

    def from_tensor_with_specials(
            self,
            tensor: torch.Tensor,
            include_unk=True,
            include_pad=True,
            include_bos=True,
            include_eos=True,
            continue_after_eos=True) -> list[I]:
        return self.vocab.idxs2tokens(
            idxs=[int(idx.item()) for idx in tensor.squeeze()],
            include_unk=include_unk,
            include_pad=include_pad,
            include_bos=include_bos,
            include_eos=include_eos,
            continue_after_eos=continue_after_eos)


class StrDataHandler(VocabDataHandler[str]):
    def __init__(
            self,
            vocab: Vocab[str],
            split_tokens: Callable[[str], list[str]],
            join_tokens: Callable[[list[str]], str],
            tokenizer: Callable[[str], list[str]] | None,
            flat_input=False,
            device: torch.device | None = None):
        super().__init__(vocab=vocab, flat_input=flat_input, device=device)
        self.vocab = vocab
        self.split_tokens = split_tokens
        self.join_tokens = join_tokens
        self.tokenizer = tokenizer
        self.device = device

    def _transform_tokens(
            self,
            tokens: list[str],
            max_length: int | None) -> torch.Tensor:

        ids = self.vocab.tokens2idxs(tokens=tokens)

        if self.vocab.eos_idx is not None:
            ids.append(self.vocab.eos_idx)

        if max_length:
            amount = min(max_length, len(ids))
            # full_ids = np.zeros((max_length), dtype=np.int32)
            # above, but fill with pad_idx instead of 0
            full_ids = np.full(
                (max_length), self.vocab.pad_idx, dtype=np.int32)
            full_ids[:amount] = ids[:amount]
        else:
            full_ids = ids

        input = torch.LongTensor(full_ids).to(self.device)

        return input

    def transform(self, input: str, max_length: int | None) -> torch.Tensor:
        vocab = self.vocab
        tokens = self.normalize_dataset_item(input)

        if max_length is not None:
            tensor = self._transform_tokens(
                tokens=tokens,
                max_length=max_length)
        else:
            token_ids = vocab.tokens2idxs(
                tokens=tokens)
            tensor = torch.cat((
                torch.tensor([vocab.bos_idx]
                             if vocab.bos_idx is not None else []).to(self.device),
                torch.tensor(token_ids).to(self.device),
                torch.tensor([vocab.eos_idx] if vocab.eos_idx is not None else []).to(
                    self.device)
            )).to(self.device)

        return tensor

    def normalize_string(self, item: str) -> str:
        return data_utils.normalize_string(item)

    def normalize_dataset_item(self, item: str) -> list[str]:
        item = item.rstrip('\n')
        result = (
            self.tokenizer(item)
            if self.tokenizer else
            self.split_tokens(data_utils.normalize_string(item)))
        return result

    def str_to_tensor(self, item: str, max_length: int) -> torch.Tensor:
        return self._transform_tokens(
            tokens=self.normalize_dataset_item(item),
            max_length=max_length)

    def idxs_to_str(self, idxs: list[int], append_eos=False) -> str:
        words = self.vocab.idxs2tokens(
            idxs=idxs,
            include_eos=append_eos)

        return self.join_tokens(words)


class LettersDataHandler(StrDataHandler):
    def __init__(
            self,
            vocab: LettersVocab,
            flat_input=False,
            device: torch.device | None = None):

        def split_tokens(text: str) -> list[str]:
            return [c for c in data_utils.unicode_to_ascii(
                text,
                allowed_letters=vocab.all_letters)]

        def join_tokens(tokens: list[str]) -> str:
            return ''.join(tokens)

        super().__init__(
            vocab=vocab,
            split_tokens=split_tokens,
            join_tokens=join_tokens,
            tokenizer=None,
            flat_input=flat_input,
            device=device)

        self.device = device


class WordsDataHandler(StrDataHandler):
    def __init__(
            self,
            vocab: Vocab[str],
            tokenizer: Callable[[str], list[str]] | None,
            flat_input=False,
            device: torch.device | None = None):

        def split_tokens(text: str) -> list[str]:
            return tokenizer(text) if tokenizer else text.split(' ')

        def join_tokens(tokens: list[str]) -> str:
            return ' '.join(tokens)

        super().__init__(
            vocab=vocab,
            split_tokens=split_tokens,
            join_tokens=join_tokens,
            tokenizer=tokenizer,
            flat_input=flat_input,
            device=device)

        self.device = device

####################################################
################### General Data ###################
####################################################


class Datasource(Generic[I]):
    def __init__(self, datasets: DatasetGroup[I]):
        self.datasets = datasets

L = TypeVar("L")
T = TypeVar("T")
L = TypeVar("L")

class LabeledDatasource(Datasource[tuple[I, T]], Generic[I, T, L]):
    def __init__(
            self,
            datasets: DatasetGroup[tuple[I, T]],
            all_labels: list[L]):

        super().__init__(datasets=datasets)

        self.all_labels = all_labels

class LabeledTensorDatasource(
    LabeledDatasource[torch.Tensor, torch.Tensor, typing.Any]
): pass

class LabeledVocabData(Generic[I]):
    def __init__(
            self,
            data: LabeledDatasource[list[I], int, str],
            vocab: Vocab[I],
            flat_input=False,
            device: torch.device | None = None):

        label_handler = LabelDataHandler(
            all_labels=data.all_labels,
            device=device)
        data_handler = VocabDataHandler(
            vocab=vocab, flat_input=flat_input, device=device)

        def _transform_to_label_tensor(item: tuple[list[I], int]) -> tuple[torch.Tensor, torch.Tensor]:
            tokens, label_idx = item
            label_tensor = self.label_handler.to_tensor(label_idx)
            input_tensor = self.data_handler.to_input_tensor(tokens)
            return label_tensor, input_tensor

        def _transform_to_target_tensor(item: tuple[list[I], int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            tokens, label_idx = item
            label_tensor = self.label_handler.to_input_tensor(label_idx)
            input_tensor = self.data_handler.to_input_tensor(tokens)
            target_tensor = self.data_handler.to_target_tensor(tokens)
            return label_tensor, input_tensor, target_tensor

        label_datasets = data.datasets.transform(_transform_to_label_tensor)
        target_datasets = data.datasets.transform(_transform_to_target_tensor)
        self.label_datasets = label_datasets
        self.target_datasets = target_datasets

        self.datasets = data.datasets
        self.vocab = vocab
        self.label_handler = label_handler
        self.data_handler = data_handler


class IOData(Generic[T, I, O]):
    def __init__(
            self,
            datasets: DatasetGroup[T],
            tensor_datasets: DatasetGroup[tuple[Tensor, Tensor]],
            collate_fn: Callable[[list[tuple[Tensor, Tensor]]], tuple[torch.Tensor, torch.Tensor]] | None,
            input_handler: FullDataHandler[I],
            output_handler: FullDataHandler[O]):

        self.datasets = datasets
        self.tensor_datasets = tensor_datasets
        self.collate_fn = collate_fn
        self.input_handler = input_handler
        self.output_handler = output_handler


class LabeledStrData(LabeledVocabData[str]):
    def __init__(
            self,
            data: LabeledDatasource[str, int, str],
            vocab_handler: StrDataHandler,
            flat_input=False,
            device: torch.device | None = None):

        vocab = vocab_handler.vocab

        def idxs_to_text(idx_list: list[int]) -> str:
            return vocab_handler.join_tokens(vocab.idxs2tokens(
                idxs=idx_list))

        def transform(batch: tuple[str, int]):
            text, label_idx = batch
            tokens = vocab_handler.split_tokens(text)
            return tokens, label_idx

        transformed_data = LabeledDatasource[list[str], int, str](
            all_labels=data.all_labels,
            datasets=data.datasets.transform(transform))

        super().__init__(
            data=transformed_data,
            vocab=vocab,
            flat_input=flat_input,
            device=device)

        self.raw_datasets = data.datasets
        self.str_vocab = vocab
        self.idxs_to_text = idxs_to_text
        self.vocab_handler = vocab_handler


class LettersLabeledData(LabeledStrData):
    def __init__(
            self,
            data: LabeledDatasource[str, int, str],
            vocab: LettersVocab,
            flat_input=False,
            device: torch.device | None = None):

        vocab_handler = LettersDataHandler(
            vocab=vocab, flat_input=flat_input, device=device)

        super().__init__(
            data=data,
            vocab_handler=vocab_handler,
            flat_input=flat_input,
            device=device)

        self.letters_vocab = vocab


class WordsLabeledData(LabeledStrData):
    def __init__(
            self,
            data: LabeledDatasource[str, int, str],
            vocab: Lang,
            tokenizer: Callable[[str], list[str]] | None = None,
            flat_input=False,
            device: torch.device | None = None):

        vocab_handler = WordsDataHandler(
            vocab=vocab, tokenizer=tokenizer, flat_input=flat_input, device=device)

        super().__init__(
            data=data,
            vocab_handler=vocab_handler,
            device=device)


class WordsIOData(IOData[tuple[str, str], list[str], list[str]]):
    def __init__(
            self,
            datasets: DatasetGroup[tuple[str, str]],
            input_lang: BareLang,
            output_lang: BareLang,
            max_length: int | None,
            input_tokenizer: Callable[[str], list[str]] | None = None,
            output_tokenizer: Callable[[str], list[str]] | None = None,
            flat_input=False,
            device: torch.device | None = None):

        input_handler = WordsDataHandler(
            vocab=input_lang, tokenizer=input_tokenizer, flat_input=flat_input, device=device)
        output_handler = WordsDataHandler(
            vocab=output_lang, tokenizer=output_tokenizer, flat_input=flat_input, device=device)

        def transform_gen(handler: WordsDataHandler):
            def transform(input: str):
                return handler.transform(input=input, max_length=max_length)

            return transform

        transform_input = transform_gen(input_handler)
        transform_output = transform_gen(output_handler)

        def transform_to_tensor(batch: tuple[str, str]):
            src_sample, tgt_sample = batch
            input_tensor = transform_input(src_sample)
            output_tensor = transform_output(tgt_sample)
            return input_tensor, output_tensor

        def collate_fn(batch: list[tuple[Tensor, Tensor]]):
            if input_lang.pad_idx is None:
                raise ValueError("Input language has no pad index")

            if output_lang.pad_idx is None:
                raise ValueError("Output language has no pad index")

            src_batch: list[Tensor] = []
            tgt_batch: list[Tensor] = []

            for src_sample, tgt_sample in batch:
                src_batch.append(src_sample)
                tgt_batch.append(tgt_sample)

            src_batch_tensor = pad_sequence(
                src_batch, padding_value=input_lang.pad_idx)
            tgt_batch_tensor = pad_sequence(
                tgt_batch, padding_value=output_lang.pad_idx)

            return src_batch_tensor, tgt_batch_tensor

        tensor_datasets = datasets.transform(transform_to_tensor)

        def add_tokens_to_vocab(item: tuple[str, str], no_input: bool, no_output: bool):
            src_sample, tgt_sample = item

            input_tokens = input_handler.normalize_dataset_item(src_sample)
            output_tokens = output_handler.normalize_dataset_item(
                tgt_sample)

            if not no_input:
                for token in input_tokens:
                    input_lang.add_token(token)

            if not no_output:
                for token in output_tokens:
                    output_lang.add_token(token)

        def fill_vocab(no_input=False, no_output=False):
            # generate all the language tokens
            # (the iteration loads the tokens from the data in the dataset)
            for item in datasets.train:
                add_tokens_to_vocab(item, no_input=no_input,
                                    no_output=no_output)

            if datasets.validation is not None:
                for item in datasets.validation:
                    add_tokens_to_vocab(
                        item, no_input=no_input, no_output=no_output)

            if datasets.test is not None:
                for item in datasets.test:
                    add_tokens_to_vocab(
                        item, no_input=no_input, no_output=no_output)

        super().__init__(
            datasets=datasets,
            tensor_datasets=tensor_datasets,
            collate_fn=collate_fn if max_length is None else None,
            input_handler=input_handler,
            output_handler=output_handler)

        self.input_handler = input_handler
        self.output_handler = output_handler

        self.raw_datasets = datasets
        self.fill_vocab = fill_vocab
        self.split_tokens = input_handler.split_tokens
        self.join_tokens = input_handler.join_tokens
        self.words_input_handler = input_handler
        self.words_output_handler = output_handler
        self.transform_input = transform_input
        self.transform_output = transform_output

####################################################
##################### Adapters #####################
####################################################


T = typing.TypeVar('T')


def _random_choice(l: list[T]) -> T:
    return l[random.randint(0, len(l) - 1)]


class LabeledStrDatasetAdapter():
    def __init__(self, data: LabeledStrData, dataset: Dataset[tuple[str, int]], n_iters: int):
        label_handler = data.label_handler
        data_handler = data.data_handler

        def _get_label_dataloader_iter(data_per_label_idx: dict[int, list[str]]):
            all_labels = label_handler.all_labels

            for _ in range(n_iters):
                label = _random_choice(all_labels)
                label_idx = all_labels.index(label)
                input = _random_choice(data_per_label_idx[label_idx])
                label_tensor = label_handler.to_tensor(label_idx)
                input_tensor = data_handler.to_input_tensor(
                    data.vocab_handler.split_tokens(input))
                yield label_tensor, input_tensor

        def _get_target_dataloader_iter(data_per_label_idx: dict[int, list[str]]):
            all_labels = data.label_handler.all_labels

            for _ in range(n_iters):
                label = _random_choice(all_labels)
                label_idx = all_labels.index(label)
                input = _random_choice(data_per_label_idx[label_idx])
                label_tensor = data.label_handler.to_input_tensor(label_idx)
                input_tensor = data.data_handler.to_input_tensor(
                    data.vocab_handler.split_tokens(input))
                target_tensor = data.data_handler.to_target_tensor(
                    data.vocab_handler.split_tokens(input))
                yield label_tensor, input_tensor, target_tensor

        _data_per_label_idx: dict[int, list[str]] | None = None
        self._data_per_label_idx = _data_per_label_idx

        def get_data_per_label_idx() -> dict[int, list[str]]:
            data_per_label_idx = self._data_per_label_idx

            if not data_per_label_idx:
                data_per_label_idx = {
                    label_idx: [
                        tokens
                        for tokens, item_label_idx
                        in dataset
                        if item_label_idx == label_idx
                    ] for label_idx, _ in enumerate(label_handler.all_labels)}
                self._data_per_label_idx = data_per_label_idx

            return data_per_label_idx

        def get_label_iter_dataloader() -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
            data_per_label_idx = get_data_per_label_idx()

            return DataLoader(
                dataset=IterDataset(lambda: _get_label_dataloader_iter(
                    data_per_label_idx=data_per_label_idx)),
                batch_size=1)

        def get_target_iter_dataloader() -> DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            data_per_label_idx = get_data_per_label_idx()

            return DataLoader(
                dataset=IterDataset(lambda: _get_target_dataloader_iter(
                    data_per_label_idx=data_per_label_idx)),
                batch_size=1)

        self.get_label_iter_dataloader = get_label_iter_dataloader
        self.get_target_iter_dataloader = get_target_iter_dataloader


class LabeledStrDataAdapter(Generic[I]):
    def __init__(self, data: LabeledStrData, split_data: SplitData, n_iters: int):
        train_adapter = LabeledStrDatasetAdapter(
            data=data,
            dataset=data.raw_datasets.train,
            n_iters=n_iters)
        validation_adapter = (
            LabeledStrDatasetAdapter(
                data=data,
                dataset=data.raw_datasets.validation,
                n_iters=math.ceil(n_iters*split_data.val_percent))
            if data.raw_datasets.validation and split_data.val_percent
            else None)
        test_adapter = (
            LabeledStrDatasetAdapter(
                data=data,
                dataset=data.raw_datasets.test,
                n_iters=math.ceil(n_iters*split_data.test_percent))
            if data.raw_datasets.test and split_data.test_percent
            else None)

        self.train = train_adapter
        self.validation = validation_adapter
        self.test = test_adapter
