from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from net import DensenetOneChannel
from preprocess import crop_resize


HEIGHT = 137
WIDTH = 236
SIZE = 128
BATCH_SIZE = 128
STATS = (0.0692, 0.2051)
MODEL = "../input/models/model_0.pth"
nworkers = 2

TEST = [
    "/kaggle/input/bengaliai-cv19/test_image_data_0.parquet",
    "/kaggle/input/bengaliai-cv19/test_image_data_1.parquet",
    "/kaggle/input/bengaliai-cv19/test_image_data_2.parquet",
    "/kaggle/input/bengaliai-cv19/test_image_data_3.parquet",
]

LABELS = "../input/train.csv"


class GraphemeDataset:
    def __init__(self, fname):
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH).astype(
            np.uint8
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.df.iloc[idx, 0]
        # normalize each image by its max val
        img = (self.data[idx] * (255.0 / self.data[idx].max())).astype(np.uint8)
        img = crop_resize(img)
        img = (img.astype(np.float32) / 255.0 - STATS[0]) / STATS[1]
        return img, name


def create_submission_csv():
    df = pd.read_csv(LABELS)
    nunique = list(df.nunique())[1:-1]
    model = DensenetOneChannel(arch=densenet121, n=nunique, pretrained=False).cuda()
    model.load_state_dict(torch.load(MODEL, map_location=torch.device("cpu")))
    model.eval()

    row_id, target = [], []
    for fname in TEST:
        ds = GraphemeDataset(fname)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=nworkers, shuffle=False)
        with torch.no_grad():
            for x, y in tqdm(dl):
                x = x.unsqueeze(1).cuda()
                p1, p2, p3 = model(x)
                p1 = p1.argmax(-1).view(-1).cpu()
                p2 = p2.argmax(-1).view(-1).cpu()
                p3 = p3.argmax(-1).view(-1).cpu()
                for idx, name in enumerate(y):
                    row_id += [
                        f"{name}_grapheme_root",
                        f"{name}_vowel_diacritic",
                        f"{name}_consonant_diacritic",
                    ]
                    target += [p1[idx].item(), p2[idx].item(), p3[idx].item()]

    sub_df = pd.DataFrame({"row_id": row_id, "target": target})
    sub_df.to_csv("submission.csv", index=False)
    sub_df.head()


if __name__ == "__main__":
    create_submission_csv()
