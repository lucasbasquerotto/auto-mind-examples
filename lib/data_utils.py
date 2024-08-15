import glob
import re
import unicodedata
from torch import Tensor
import torch

def find_files(path: str): return glob.glob(path)

def unicode_to_ascii(s: str, allowed_letters: str | None = None):
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and (allowed_letters is None or c in allowed_letters))

def normalize_string(s: str, allowed_letters: str | None = None):
    s = unicode_to_ascii(s.lower().strip(), allowed_letters=allowed_letters)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()

def read_lines(filename: str):
    # Read a file and split into lines
    with open(filename, encoding='utf-8') as some_file:
        return [line.strip() for line in some_file]

def offsetify(tensors: list[Tensor]):
    # compute the offsets by accumulating the tensor of sequence lengths
    o = [0] + [len(t) for t in tensors]
    offset = torch.tensor(o[:-1]).cumsum(dim=0)
    return offset
