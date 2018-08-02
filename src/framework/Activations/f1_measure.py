# Global imports
import torch
import torch.nn as nn


class F1Loss(nn.Module):

    def __init__(self):
        super(F1Loss, self).__init__()

    def precision_loss(self, y_true, y_pred):
        true_positives = torch.sum(torch.clamp(y_pred * y_true, 0, 1))
        false_positives = torch.sum(y_pred * torch.clamp(1 - y_true, 0, 1))
        precision = true_positives / (true_positives + false_positives + 1e-8)

        return precision

    def recall_loss(self, y_true, y_pred):
        true_positives = torch.sum(torch.clamp(y_pred * y_true, 0, 1))
        false_negatives = torch.sum(torch.clamp((1 - y_pred) * y_true, 0, 1))
        recall = true_positives / (true_positives + false_negatives + 1e-8)

        return recall

    def fbeta_score_loss(self, y_true, y_pred, beta=1.):
        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')

        # If there are no true positives, fix the F score at 0 like sklearn.
        if torch.sum(torch.clamp(y_true, 0, 1)) == 0:
            return 0

        p = self.precision_loss(y_true, y_pred)
        r = self.recall_loss(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / (bb * p + r + 1e-8)

        return fbeta_score

    def f1measure_loss(self, y_true, y_pred):
        return -1 * self.fbeta_score_loss(y_true, y_pred, beta=1)

    def forward(self, input, target):
        return self.f1measure_loss(target, input)
