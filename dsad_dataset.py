# dsad_dataset.py
import os
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader


class AbdominalWallDataset(Dataset):
    """
    Loads a small number of images from:
        DSAD/abdominal_wall/03/imageXX.png
    and gives them label 0 (abdominal_wall).
    """

    def __init__(self,
                 root_dir="../DSAD/abdominal_wall/03",
                 max_images=20,
                 img_size=512):
        self.root_dir = root_dir

        # grab only files like image00.png, image01.png, ...
        files = [
            f for f in os.listdir(root_dir)
            if f.startswith("image") and f.endswith(".png")
        ]
        files.sort()
        self.files = files[:max_images]   # <-- only use first N images

        self.transform = T.Compose([
            T.Grayscale(),                # 1 channel
            T.Resize((img_size, img_size)),
            T.ToTensor(),                 # [1,H,W], values in [0,1]
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.root_dir, fname)

        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        label = 0     # abdominal_wall class id
        return img, label


# create dataset + loader that train.py can import
dataset = AbdominalWallDataset(max_images=20)
loader = DataLoader(dataset, batch_size=4, shuffle=True)
