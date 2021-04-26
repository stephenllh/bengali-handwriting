from functools import partial
from torch import nn
from torch.nn import functional as F


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, reduction="mean"):
        x1, x2, x3 = input
        x1, x2, x3 = x1.float(), x2.float(), x3.float()
        y = target.long()
        return (
            0.7 * F.cross_entropy(x1, y[:, 0], reduction=reduction)
            + 0.1 * F.cross_entropy(x2, y[:, 1], reduction=reduction)
            + 0.2 * F.cross_entropy(x3, y[:, 2], reduction=reduction)
        )


class MixUpLoss(nn.Module):
    "Adapt the loss function `crit` to go with mixup."

    def __init__(self, crit, reduction="mean"):
        super().__init__()
        if hasattr(crit, "reduction"):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, "reduction", "none")
        else:
            self.crit = partial(crit, reduction="none")
            self.old_crit = crit
        self.reduction = reduction

    def forward(self, output, target):
        if len(target.shape) == 2 and target.shape[1] == 7:
            loss1, loss2 = self.crit(output, target[:, 0:3].long()), self.crit(
                output, target[:, 3:6].long()
            )
            d = loss1 * target[:, -1] + loss2 * (1 - target[:, -1])
        else:
            d = self.crit(output, target)
        if self.reduction == "mean":
            return d.mean()
        elif self.reduction == "sum":
            return d.sum()
        return d

    def get_old(self):
        if hasattr(self, "old_crit"):
            return self.old_crit
        elif hasattr(self, "old_red"):
            setattr(self.crit, "reduction", self.old_red)
            return self.crit
