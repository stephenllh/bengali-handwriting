import pandas as pd
from functools import partial
from csvlogger import CSVLogger
from torchvision.models import densenet121
from fastai.vision.data import ImageList
from fastai.vision.transform import get_transforms
from fastai.learner import Learner
from radam import Over9000
from net import DensenetOneChannel
from loss import CombinedLoss
from callback import MetricIndex, MetricTotal, SaveModelCallback, MixUpCallback
from util import seed_everything


IMAGE_SIZE = 128
BATCH_SIZE = 128
N_FOLDS = 4
FOLD = 0
SEED = 2019
TRAIN = "../input/grapheme-imgs-128x128/"
LABELS = "../input/bengaliai-cv19/train.csv"


def train():
    seed_everything(0)
    df = pd.read_csv(LABELS)

    stats = ([0.0692], [0.2051])
    data = (
        ImageList.from_df(
            df, path=".", folder=TRAIN, suffix=".png", cols="image_id", convert_mode="L"
        )
        .split_by_idx(range(FOLD * len(df) // N_FOLDS, (FOLD + 1) * len(df) // N_FOLDS))
        .label_from_df(cols=["grapheme_root", "vowel_diacritic", "consonant_diacritic"])
        .transform(
            get_transforms(do_flip=False, max_warp=0.1),
            size=IMAGE_SIZE,
            padding_mode="zeros",
        )
        .databunch(BATCH_SIZE=BATCH_SIZE)
    ).normalize(stats)

    nunique = list(df.nunique())[1:-1]
    model = DensenetOneChannel(arch=densenet121, n=nunique)

    MetricGrapheme = partial(MetricIndex, 0)
    MetricVowel = partial(MetricIndex, 1)
    MetricConsonant = partial(MetricIndex, 2)
    learn = Learner(
        data,
        model,
        loss_func=CombinedLoss(),
        opt_func=Over9000,
        metrics=[MetricGrapheme(), MetricVowel(), MetricConsonant(), MetricTotal()],
    )

    logger = CSVLogger(learn, f"log{FOLD}")
    learn.clip_grad = 1.0
    learn.split([model.head1])
    learn.unfreeze()

    learn.fit_one_cycle(
        cyc_len=32,  # essentially number of epochs
        max_lr=slice(2e-3, 1e-2),
        wd=[1e-3, 0.1e-1],
        pct_start=0.0,
        div_factor=100,
        callbacks=[
            logger,
            SaveModelCallback(
                learn, monitor="metric_tot", mode="max", name=f"model_{FOLD}"
            ),
            MixUpCallback(learn),
        ],
    )


if __name__ == "__main__":
    train()
