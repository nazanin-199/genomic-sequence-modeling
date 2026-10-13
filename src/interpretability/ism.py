import torch
import numpy as np
from typing import Tuple
from ..data.dataset import one_hot_encode_batch

@torch.no_grad()
def batched_saturation_ism(
    model: torch.nn.Module,
    seq: str,
    device: torch.device,
    batch_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes saturation In Silico Mutagenesis (ISM) across forward logits.
    
    Returns:
        ism_profiles: (L, num_tasks) attribution array
        base_logits: (num_tasks,) unmutated logit array
    """
    model.eval()
    L = len(seq)
    bases = ['A', 'C', 'G', 'T']
    
    x_orig = one_hot_encode_batch([seq]).to(device)
    base_logits = model(x_orig)[0].cpu().numpy()
    
    mutants = []
    positions = []
    for t in range(L):
        orig_b = seq[t]
        for b in bases:
            if b != orig_b and orig_b in bases:
                mut_seq = list(seq)
                mut_seq[t] = b
                mutants.append(''.join(mut_seq))
                positions.append(t)
                
    mut_logits = []
    for i in range(0, len(mutants), batch_size):
        xb = one_hot_encode_batch(mutants[i:i + batch_size]).to(device)
        mut_logits.append(model(xb).cpu().numpy())
    mut_logits = np.vstack(mut_logits)
    
    num_tasks = base_logits.shape[0]
    ism_profiles = np.zeros((L, num_tasks), dtype=np.float32)
    counts = np.zeros(L, dtype=np.float32)
    
    for idx, pos in enumerate(positions):
        for task in range(num_tasks):
            drop = max(0.0, float(base_logits[task] - mut_logits[idx, task]))
            ism_profiles[pos, task] += drop
        counts[pos] += 1.0
        
    valid = counts > 0
    ism_profiles[valid] /= counts[valid, None]
    return ism_profiles, base_logits
