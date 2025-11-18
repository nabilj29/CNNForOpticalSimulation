import torch
import torch.nn as nn
import torchoptics
import torch.nn.functional as F
from torchoptics.elements import Lens, Modulator
from torchoptics import Field, System

class Optical4f(nn.Module):
    def __init__(self, kernel, shape=512, spacing=10e-6, wavelength=532e-9, f=200e-3):
        super().__init__()

        self.shape = shape
        self.f = f

        # Set global optical parameters
        torchoptics.set_default_spacing(spacing)
        torchoptics.set_default_wavelength(wavelength)

        # === Build radial Gaussian plane wave ===
        x = torch.arange(0, shape, dtype=torch.float32) - shape // 2
        y = torch.arange(0, shape, dtype=torch.float32) - shape // 2
        X, Y = torch.meshgrid(x, y)
        r = torch.sqrt(X**2 + Y**2)
        sigma = torch.max(r) / torch.sqrt(-2 * torch.log(torch.tensor(0.85)))

        gaussian = torch.exp(-(r**2) / (2 * sigma**2))
        self.register_buffer("gaussian_wave", gaussian.to(torch.complex64))

        # === Create Fourier-plane filter modulator ===
        self.kernel_modulator = self.kernel_to_modulator(kernel, shape)

        # === Optical 4f System ===
        self.system = System(
            Lens(shape * 2, f, z=1 * f),
            self.kernel_modulator,
            Lens(shape * 2, f, z=3 * f),
        )

    def kernel_to_modulator(self, kernel, shape):
        """Convert kernel → Fourier-plane amplitude modulator."""
        H, W = shape, shape
        pad_h = (H - kernel.shape[0]) // 2
        pad_w = (W - kernel.shape[1]) // 2
        padded = F.pad(kernel, (pad_w, pad_w, pad_h, pad_h))

        Fk = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded)))
        amp = torch.abs(Fk)
        amp = amp / amp.max()

        return Modulator(amp)

    def forward(self, img):
        """
        img: Tensor [B,1,H,W] in [0,1]
        """
        B = img.shape[0]

        outputs = []
        for i in range(B):
            # Construct input field: plane wave * image mask
            mask = img[i,0,:,:].to(torch.complex64)
            field = Field(self.gaussian_wave * mask)

            # Propagate through system
            out_field = self.system(field)

            # Camera intensity
            intensity = torch.abs(out_field.data) ** 2

            # Normalize to [0,1]
            intensity = intensity / intensity.max()

            outputs.append(intensity.unsqueeze(0))

        return torch.stack(outputs, dim=0)  # → [B,1,H,W]
