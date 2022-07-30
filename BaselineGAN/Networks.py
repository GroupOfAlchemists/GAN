import math
import torch
import torch.nn as nn
from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler, InplaceUpsampler, InplaceDownsampler
from .FusedOperators import BiasedActivation

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class GeneratorBlock(nn.Module):
    def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
        super(GeneratorBlock, self).__init__()
        
        CompressedChannels = InputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedChannels)
        self.NonLinearity2 = BiasedActivation(CompressedChannels)
        self.NonLinearity3 = BiasedActivation(InputChannels)
        
    def forward(self, x, ActivatedFeatures):
        y = self.LinearLayer1(ActivatedFeatures)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        y = x + y
          
        return y, self.NonLinearity3(y)

class DiscriminatorBlock(nn.Module):
    def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
        super(DiscriminatorBlock, self).__init__()
        
        CompressedChannels = InputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(InputChannels)
        self.NonLinearity2 = BiasedActivation(CompressedChannels)
        self.NonLinearity3 = BiasedActivation(CompressedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y))
        y = self.LinearLayer3(self.NonLinearity3(y))
        
        return x + y

class GeneratorUpsampleBlock(nn.Module):
    def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter):
        super(GeneratorUpsampleBlock, self).__init__()
        
        CompressedInputChannels = InputChannels // CompressionFactor
        CompressedOutputChannels = OutputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedInputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedInputChannels, CompressedOutputChannels * 4, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedOutputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedInputChannels)
        self.NonLinearity2 = BiasedActivation(CompressedOutputChannels)
        self.NonLinearity3 = BiasedActivation(OutputChannels)
        
        self.MainResampler = InplaceUpsampler(ResamplingFilter)
        self.ShortcutResampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x, ActivatedFeatures):
        if hasattr(self, 'ShortcutLayer'):
            x = self.ShortcutLayer(x)
        
        y = self.LinearLayer1(ActivatedFeatures)
        y = self.MainResampler(self.LinearLayer2(self.NonLinearity1(y)))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        y = self.ShortcutResampler(x) + y
        
        return y, self.NonLinearity3(y)

class DiscriminatorDownsampleBlock(nn.Module):
    def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter):
        super(DiscriminatorDownsampleBlock, self).__init__()
        
        CompressedInputChannels = InputChannels // CompressionFactor
        CompressedOutputChannels = OutputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedInputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedInputChannels * 4, CompressedOutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedOutputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(InputChannels)
        self.NonLinearity2 = BiasedActivation(CompressedInputChannels)
        self.NonLinearity3 = BiasedActivation(CompressedOutputChannels)
        
        self.MainResampler = InplaceDownsampler(ResamplingFilter)
        self.ShortcutResampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.MainResampler(self.NonLinearity2(y)))
        y = self.LinearLayer3(self.NonLinearity3(y))
        
        x = self.ShortcutResampler(x)
        if hasattr(self, 'ShortcutLayer'):
            x = self.ShortcutLayer(x)

        return x + y
     
class GeneratorInitialBlock(nn.Module):
    def __init__(self, InputUnits, OutputChannels, CompressionFactor, *_, **__):
        super(GeneratorInitialBlock, self).__init__()
        
        CompressedInputUnits = InputUnits // CompressionFactor
        CompressedOutputChannels = OutputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Linear(InputUnits, CompressedInputUnits, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(CompressedInputUnits, CompressedOutputChannels * 16, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedOutputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedInputUnits)
        self.NonLinearity2 = BiasedActivation(CompressedOutputChannels)
        self.NonLinearity3 = BiasedActivation(OutputChannels)
        
        self.Basis = nn.Parameter(torch.empty((OutputChannels, 4, 4)))
        self.Basis.data.normal_(0, 1)
        
        if InputUnits != OutputChannels:
            self.ShortcutLayer = MSRInitializer(nn.Linear(InputUnits, OutputChannels, bias=False))

    def forward(self, x, ActivatedFeatures):
        if hasattr(self, 'ShortcutLayer'):
            x = self.ShortcutLayer(x)
        
        y = self.LinearLayer1(ActivatedFeatures)
        y = self.LinearLayer2(self.NonLinearity1(y)).view(x.shape[0], -1, 4, 4)
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        x = self.Basis.view(1, -1, 4, 4) * x.view(x.shape[0], -1, 1, 1)
        y = x + y
        
        return y, self.NonLinearity3(y)

class DiscriminatorFinalBlock(nn.Module):
    def __init__(self, InputChannels, OutputUnits, CompressionFactor, *_, **__):
        super(DiscriminatorFinalBlock, self).__init__()
        
        CompressedInputChannels = InputChannels // CompressionFactor
        CompressedOutputUnits = OutputUnits // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedInputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(CompressedInputChannels * 16, CompressedOutputUnits, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Linear(CompressedOutputUnits, OutputUnits, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(InputChannels)
        self.NonLinearity2 = BiasedActivation(CompressedInputChannels)
        self.NonLinearity3 = BiasedActivation(CompressedOutputUnits)
        
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        
        if InputChannels != OutputUnits:
            self.ShortcutLayer = MSRInitializer(nn.Linear(InputChannels, OutputUnits, bias=False))
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y).view(x.shape[0], -1))
        y = self.LinearLayer3(self.NonLinearity3(y))
        
        x = self.Basis(x).view(x.shape[0], -1)
        if hasattr(self, 'ShortcutLayer'):
            x = self.ShortcutLayer(x)

        return x + y
     
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, TransitionBlock, CompressionFactor, ReceptiveField, ResamplingFilter=None):
        super(GeneratorStage, self).__init__()
        
        self.BlockList = nn.ModuleList([TransitionBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter)] + [GeneratorBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks - 1)])
        
    def forward(self, x, ActivatedFeatures):
        for Block in self.BlockList:
            x, ActivatedFeatures = Block(x, ActivatedFeatures)
        return x, ActivatedFeatures
        
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, TransitionBlock, CompressionFactor, ReceptiveField, ResamplingFilter=None):
        super(DiscriminatorStage, self).__init__()

        self.BlockList = nn.ModuleList([DiscriminatorBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks - 1)] + [TransitionBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter)])
      
    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        return x
        
class FullyConnectedBlock(nn.Module):
    def __init__(self, InputUnits, CompressionFactor):
        super(FullyConnectedBlock, self).__init__()
        
        CompressedUnits = InputUnits // CompressionFactor

        self.LinearLayer1 = MSRInitializer(nn.Linear(InputUnits, CompressedUnits, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(CompressedUnits, CompressedUnits, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Linear(CompressedUnits, InputUnits, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(InputUnits)
        self.NonLinearity2 = BiasedActivation(CompressedUnits)
        self.NonLinearity3 = BiasedActivation(CompressedUnits)
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y))
        y = self.LinearLayer3(self.NonLinearity3(y))
        
        return x + y
           
class NoiseMapping(nn.Module):
    def __init__(self, NoiseDimension, LatentDimension, NumberOfBlocks, CompressionFactor):
        super(NoiseMapping, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Linear(NoiseDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
        self.NonLinearity = BiasedActivation(LatentDimension)

        self.BlockList = nn.ModuleList([FullyConnectedBlock(LatentDimension, CompressionFactor) for _ in range(NumberOfBlocks)])
        
    def forward(self, z):
        x = self.LinearLayer(z)
        
        for Block in self.BlockList:
            x = Block(x)
        return x, self.NonLinearity(x)
      
class BinaryClassifier(nn.Module):
    def __init__(self, LatentDimension, NumberOfBlocks, CompressionFactor):
        super(BinaryClassifier, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Linear(LatentDimension, 1))
        self.NonLinearity = BiasedActivation(LatentDimension)

        self.BlockList = nn.ModuleList([FullyConnectedBlock(LatentDimension, CompressionFactor) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        return self.LinearLayer(self.NonLinearity(x)).view(x.shape[0])
        
def ToRGB(InputChannels, ResidualComponent=False):
    return MSRInitializer(nn.Conv2d(InputChannels, 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0 if ResidualComponent else 1)

class Generator(nn.Module):
    def __init__(self, NoiseDimension=512, LatentDimension=1024, LatentMappingBlocks=4, StageWidths=[1024, 1024, 1024, 1024, 1024, 1024, 512, 256, 128], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        MainLayers = [GeneratorStage(LatentDimension, StageWidths[0], BlocksPerStage[0], GeneratorInitialBlock, CompressionFactor, ReceptiveField)]
        AggregationLayers = [ToRGB(StageWidths[0])]
        for x in range(len(StageWidths) - 1):
            MainLayers += [GeneratorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x + 1], GeneratorUpsampleBlock, CompressionFactor, ReceptiveField, ResamplingFilter)]
            AggregationLayers += [ToRGB(StageWidths[x + 1], ResidualComponent=True)]
        
        self.LatentLayer = NoiseMapping(NoiseDimension, LatentDimension, LatentMappingBlocks, CompressionFactor)
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayers = nn.ModuleList(AggregationLayers)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)

    def forward(self, z):
        x, ActivatedFeatures = self.MainLayers[0](*self.LatentLayer(z))
        AggregatedOutput = self.AggregationLayers[0](ActivatedFeatures)
        
        for Layer, Aggregate in zip(self.MainLayers[1:], self.AggregationLayers[1:]):
            x, ActivatedFeatures = Layer(x, ActivatedFeatures)
            AggregatedOutput = self.Resampler(AggregatedOutput) + Aggregate(ActivatedFeatures)
        
        return AggregatedOutput

class Discriminator(nn.Module):
    def __init__(self, LatentDimension=1024, LatentMappingBlocks=4, StageWidths=[128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        MainLayers = []
        for x in range(len(StageWidths) - 1):
            MainLayers += [DiscriminatorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x], DiscriminatorDownsampleBlock, CompressionFactor, ReceptiveField, ResamplingFilter)]
        MainLayers += [DiscriminatorStage(StageWidths[-1], LatentDimension, BlocksPerStage[-1], DiscriminatorFinalBlock, CompressionFactor, ReceptiveField)]
        
        self.FromRGB = MSRInitializer(nn.Conv2d(3, StageWidths[0], kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.MainLayers = nn.ModuleList(MainLayers)
        self.CriticLayer = BinaryClassifier(LatentDimension, LatentMappingBlocks, CompressionFactor)
        
    def forward(self, x):
        x = self.FromRGB(x)

        for Layer in self.MainLayers:
            x = Layer(x)
        
        return self.CriticLayer(x)