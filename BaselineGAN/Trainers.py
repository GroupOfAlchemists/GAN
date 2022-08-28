import torch
import torch.nn as nn
import copy

def ZeroCenteredGradientPenalty(Samples, Critics):
    Gradient, = torch.autograd.grad(outputs=Critics.sum(), inputs=Samples, create_graph=True, only_inputs=True)
    return 0.5 * Gradient.square().sum([1, 2, 3])

class AdversarialTraining:
    def __init__(self, Generator, Discriminator):
        self.Generator = Generator
        self.Discriminator = Discriminator
        self.NoiseDimension = Generator.FeatureLayer.LinearLayer.weight.shape[1]
    
    def AccumulateDiscriminatorGradients(self, RealSamples, Gamma, Scale=1):
        RealSamples.requires_grad = True
        
        Noise = torch.randn(RealSamples.shape[0], self.NoiseDimension, device=RealSamples.device)
        FakeSamples = self.Generator(Noise)
        FakeSamples.requires_grad = True
        
        RealLogits = self.Discriminator(RealSamples)
        FakeLogits = self.Discriminator(FakeSamples)
        
        R1Penalty = ZeroCenteredGradientPenalty(RealSamples, RealLogits)
        R2Penalty = ZeroCenteredGradientPenalty(FakeSamples, FakeLogits)
        
        RelativisticLogits = RealLogits - FakeLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        DiscriminatorLoss = AdversarialLoss + Gamma * (R1Penalty + R2Penalty)
        (Scale * DiscriminatorLoss.mean()).backward()
        
        return [x.detach() for x in [DiscriminatorLoss, AdversarialLoss, RelativisticLogits, R1Penalty, R2Penalty]]
    
    def AccumulateGeneratorGradients(self, RealSamples, Scale=1):
        Noise = torch.randn(RealSamples.shape[0], self.NoiseDimension, device=RealSamples.device)
        FakeSamples = self.Generator(Noise)
        
        FakeLogits = self.Discriminator(FakeSamples)
        RealLogits = self.Discriminator(RealSamples)
        
        RelativisticLogits = FakeLogits - RealLogits
        AdversarialLoss = nn.functional.softplus(-RelativisticLogits)
        
        (Scale * AdversarialLoss.mean()).backward()
        
        return [x.detach() for x in [AdversarialLoss, RelativisticLogits]]
    
class ExponentialMovingAverage:
    def __init__(self, ReferenceModel):
        self.ReferenceModel = ReferenceModel
        self.Model = copy.deepcopy(ReferenceModel).eval()
    
    def Step(self, Beta):
        with torch.no_grad():
            for ParameterEMA, CurrentParameter in zip(self.Model.parameters(), self.ReferenceModel.parameters()):
                ParameterEMA.copy_(CurrentParameter.lerp(ParameterEMA, Beta))
            for BufferEMA, CurrentBuffer in zip(self.Model.buffers(), self.ReferenceModel.buffers()):
                BufferEMA.copy_(CurrentBuffer)