# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import torch
from torch_utils import training_stats

from BaselineGAN.Trainers import AdversarialTraining

#----------------------------------------------------------------------------

class Loss:
    def __init__(self, G, D, gamma=10):
        super().__init__()
        self.gan                = AdversarialTraining(Generator=G, Discriminator=D)
        self.gamma              = gamma

    def accumulate_gradients(self, phase, real_img, scale):
        assert phase in ['G', 'D']
        
        if phase == 'G':
            with torch.autograd.profiler.record_function('G_accum_step'):
                AdversarialLoss, RelativisticLogits = self.gan.AccumulateGeneratorGradients(real_img, Scale=scale)
                training_stats.report('Loss/scores/fake', RelativisticLogits)
                training_stats.report('Loss/G/loss', AdversarialLoss)

        if phase == 'D':
            with torch.autograd.profiler.record_function('D_accum_step'):
                DiscriminatorLoss, AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty = self.gan.AccumulateDiscriminatorGradients(real_img, Gamma=self.gamma, Scale=scale)
                training_stats.report('Loss/scores/real', RelativisticLogits)
                training_stats.report('Loss/D/loss', AdversarialLoss)
                training_stats.report('Loss/D/regularized_loss', DiscriminatorLoss)
                training_stats.report('Loss/r1_penalty', R1Penalty)
                training_stats.report('Loss/r2_penalty', R2Penalty)
                training_stats.report('Loss/D/r1_reg', self.gamma * R1Penalty)
                training_stats.report('Loss/D/r2_reg', self.gamma * R2Penalty)

#----------------------------------------------------------------------------
