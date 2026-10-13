import torch
from src.models.transformer import MultiTaskCNNTransformer
from src.models.cnn import MultiTaskGenomicCNN
from src.models.layers import ReverseComplementConv1d

def test_rc_conv_symmetry():
    layer = ReverseComplementConv1d(4, 16, kernel_size=5, padding=2)
    x = torch.randn(2, 4, 100)
    # Flip both sequence length and reverse-complement channels [A,C,G,T] -> [T,G,C,A]
    x_rc = torch.flip(x, dims=[1, 2])
    out = layer(x)
    out_rc = layer(x_rc)
    # Layer output from forward and RC inputs must have identical magnitude distribution
    assert torch.isclose(out.mean(), out_rc.mean(), atol=1e-4)

def test_transformer_dimensions():
    model = MultiTaskCNNTransformer(num_tasks=2)
    x = torch.randn(4, 4, 500)
    out = model(x)
    assert out.shape == (4, 2)
