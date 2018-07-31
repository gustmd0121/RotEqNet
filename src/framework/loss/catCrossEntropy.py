import torch.nn as nn

def Cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=0)
    return nn.CrossEntropyLoss()(input, labels)
