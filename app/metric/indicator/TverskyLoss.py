import numpy as np
import torch
from torch import nn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(predict, label, axes=None):
    if (axes == None):
        axes = tuple(range(2, len(predict.size())))
    shp_x = predict.shape
    shp_y = label.shape
    with torch.no_grad():
        if (len(shp_x) != len(shp_y)):
            label = label.view(shp_y[0], 1, *shp_y[1:])
        if all([i == j for i, j in zip(predict.shape, label.shape)]):
            # if this is the case then label is probably already a one hot encoding
            y_onehot = label
        else:
            label = label.long()
            y_onehot = torch.zeros(shp_x)
            if predict.device.type == "cuda":
                y_onehot = y_onehot.cuda(predict.device.index)
            y_onehot.scatter_(1, label, 1)
    tp = predict * y_onehot
    fn = (1 - predict) * y_onehot
    fp = (1 - y_onehot) * predict

    tp = sum_tensor(tp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    return tp, fn, fp


class TverskyLoss(nn.Module):
    def __init__(self, batch_dice=False, smooth=1., alpha=0.7):
        super(TverskyLoss, self).__init__()
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, x, y):  # x is predict y is label
        shp_x = x.shape
        if self.batch_dice:
             axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))
        tp, fn, fp = get_tp_fp_fn(x, y, axes)
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + (1 - self.alpha) * fp + self.smooth)
        dc = 1 - tversky
        return dc


if __name__ == '__main__':
    predict = torch.rand(size=(2, 3, 16, 16, 16))
    label = torch.ones(size=(2, 3, 16, 16, 16))
    print(predict.shape, label.shape)
    tp, fn, fp = get_tp_fp_fn(predict, label, axes=None)
    print(tp, tp.shape)
    dc=TverskyLoss(batch_dice=True)(predict,label)
    print(dc)