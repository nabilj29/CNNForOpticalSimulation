import os
import torch
import torch.nn as nn
from onnClass import ONNClassifier        # wherever you defined it
from dsad_dataset import loader           # your DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Simple fixed edge-detection kernel (Laplacian) for the optical 4f module
edge_kernel = torch.tensor(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, 8.0, -1.0],
        [-1.0, -1.0, -1.0],
    ],
    dtype=torch.float32,
)

model = ONNClassifier(kernel=edge_kernel, num_classes=12).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for img, label in loader:
        # img: [B,1,H,W] float in [0,1]
        img = img.to(device).float()
        label = label.to(device).long()   # CrossEntropyLoss needs Long

        out = model(img)                 # [B,num_classes]
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch", epoch, "loss =", loss.item())

# Persist the trained weights so they can be reloaded later
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "onn_classifier.pt")
torch.save(
    {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch + 1,
    },
    checkpoint_path,
)
print(f"Saved checkpoint to {checkpoint_path}")
