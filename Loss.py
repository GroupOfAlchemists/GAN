import torch
import torch.nn as nn

def ZeroCenteredGradientPenalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True, only_inputs=True)
    return 0.5 * Gradient.square().sum([1,2,3]).mean()

def RelativisticLoss(PositiveCritics, NegativeCritics):
    return nn.functional.binary_cross_entropy_with_logits(PositiveCritics - NegativeCritics, torch.ones_like(PositiveCritics))
