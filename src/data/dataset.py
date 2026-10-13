import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

NUC_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
COMP_TRANS = str.maketrans('ACGTN', 'TGCAN')

def one_hot_encode_batch(sequences: List[str]) -> torch.Tensor:
    """Encodes a list of DNA sequences of length L into (N, 4, L) tensor."""
    n_seq = len(sequences)
    seq_len = len(sequences[0])
    out = np.zeros((n_seq, 4, seq_len), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for pos, char in enumerate(seq):
            if char in NUC_TO_IDX:
                out[i, NUC_TO_IDX[char], pos] = 1.0
            else:
                out[i, :, pos] = 0.25
    return torch.from_numpy(out)

def reverse_complement(seq: str) -> str:
    """Returns the biological reverse complement of a DNA sequence."""
    return seq.translate(COMP_TRANS)[::-1]

class MultiLineageGenomicDataset(Dataset):
    """Dataset providing DNA sequences and multi-lineage label tensors."""
    def __init__(self, sequences: List[str], labels: np.ndarray):
        assert len(sequences) == len(labels), "Length mismatch between sequences and labels"
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]

def collate_genomic_batch(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    seqs, labels = zip(*batch)
    return one_hot_encode_batch(list(seqs)), torch.stack(labels)
