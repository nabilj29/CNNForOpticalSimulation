import importlib
import torch
import torch.nn as nn

# 4fout.py cannot be imported with a normal statement because its name
# begins with a digit. Load it dynamically instead.
_fourf_mod = importlib.import_module("4fout")
Optical4f = getattr(_fourf_mod, "Optical4f")


class ONNClassifier(nn.Module):
    """
    Simple classifier that pipes the input through the optical 4f system
    and then applies a lightweight MLP head on pooled intensities.
    """

    def __init__(
        self,
        kernel: torch.Tensor,
        num_classes: int = 12,
        image_size: int = 512,
        pooled_size: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()

        if not torch.is_tensor(kernel):
            kernel = torch.tensor(kernel, dtype=torch.float32)
        else:
            kernel = kernel.to(dtype=torch.float32)

        self.optical = Optical4f(kernel=kernel, shape=image_size)
        self.pool = nn.AdaptiveAvgPool2d((pooled_size, pooled_size))

        flattened = pooled_size * pooled_size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        optical_out = self.optical(x)
        pooled = self.pool(optical_out).float()
        logits = self.classifier(pooled)
        return logits
