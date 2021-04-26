import cv2
from tqdm import tqdm
import zipfile
import pandas as pd
import numpy as np


HEIGHT = 137
WIDTH = 236
SIZE = 128

TRAIN = [
    "../input/train_image_data_0.parquet",
    "../input/train_image_data_1.parquet",
    "../input/train_image_data_2.parquet",
    "../input/train_image_data_3.parquet",
]

OUT_TRAIN = "train.zip"


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def crop_resize(img0, size=SIZE, pad=16):
    """
    Crop a box around pixels large than the threshold.
    Some images contain line at the sides.
    """
    ymin, ymax, xmin, xmax = bbox(img0[5:-5, 5:-5] > 80)

    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax, xmin:xmax]

    # Remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax - xmin, ymax - ymin
    length = max(lx, ly) + pad

    # Ensure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((length - ly) // 2,), ((length - lx) // 2,)], mode="constant")

    return cv2.resize(img, (size, size))


def preprocess_and_save():
    x_tot, x2_tot = [], []
    with zipfile.ZipFile(OUT_TRAIN, "w") as img_out:
        for fname in TRAIN:
            df = pd.read_parquet(fname)

            # The input is inverted
            data = 255 - df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
            data = data.astype(np.uint8)

            for idx in tqdm(range(len(df))):
                name = df.iloc[idx, 0]

                # Normalize each image by its max val
                img = (data[idx] * (255.0 / data[idx].max())).astype(np.uint8)
                img = crop_resize(img)

                x_tot.append((img / 255.0).mean())
                x2_tot.append(((img / 255.0) ** 2).mean())
                img = cv2.imencode(".png", img)[1]
                img_out.writestr(name + ".png", img)


if __name__ == "__main__":
    preprocess_and_save()
