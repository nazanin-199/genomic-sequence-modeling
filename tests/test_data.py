import torch
from src.data.dataset import one_hot_encode_batch, reverse_complement

def test_one_hot_encoding():
    seq = ["ACGT"]
    tensor = one_hot_encode_batch(seq)
    assert tensor.shape == (1, 4, 4)
    # Check identity diagonal
    assert torch.allclose(tensor[0], torch.eye(4))

def test_reverse_complement():
    assert reverse_complement("TGGCCACCAGGGGGCGCTA") == "TAGCGCCCCCTGGTGGCCA"
