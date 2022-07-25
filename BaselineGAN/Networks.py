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
        
    def forward(self, x, ActivationMaps):
        y = self.LinearLayer1(ActivationMaps)
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

    def forward(self, x, ActivationMaps):
        if hasattr(self, 'ShortcutLayer'):
            x = self.ShortcutLayer(x)
        
        y = self.LinearLayer1(ActivationMaps)
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
     
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Blocks, CompressionFactor, ReceptiveField, ResamplingFilter):
        super(GeneratorStage, self).__init__()
        
        self.BlockList = nn.ModuleList([GeneratorUpsampleBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter)] + [GeneratorBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)])
        
    def forward(self, x, ActivationMaps):
        for Block in self.BlockList:
            x, ActivationMaps = Block(x, ActivationMaps)
        return x, ActivationMaps
        
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, Blocks, CompressionFactor, ReceptiveField, ResamplingFilter):
        super(DiscriminatorStage, self).__init__()

        self.BlockList = nn.ModuleList([DiscriminatorBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)] + [DiscriminatorDownsampleBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter)])
      
    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        return x
        
class GeneratorPrologStage(nn.Module):
    def __init__(self, LatentDimension, OutputChannels, Blocks, CompressionFactor, ReceptiveField):
        super(GeneratorPrologStage, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty((OutputChannels, 4, 4)))
        self.Basis.data.normal_(0, BiasedActivation.Gain)
        
        self.LinearLayer = MSRInitializer(nn.Linear(LatentDimension, OutputChannels, bias=False))
        self.NonLinearity = BiasedActivation(OutputChannels)
        
        self.BlockList = nn.ModuleList([GeneratorBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)])
        
    def forward(self, w):
        x = self.LinearLayer(w).view(w.shape[0], -1, 1, 1)
        x = self.Basis.view(1, -1, 4, 4) * x
        ActivationMaps = self.NonLinearity(x)
        
        for Block in self.BlockList:
            x, ActivationMaps = Block(x, ActivationMaps)
        return x, ActivationMaps
     
class DiscriminatorEpilogStage(nn.Module):
    def __init__(self, InputChannels, LatentDimension, Blocks, CompressionFactor, ReceptiveField):
        super(DiscriminatorEpilogStage, self).__init__()
        
        self.BlockList = nn.ModuleList([DiscriminatorBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)])
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        self.LinearLayer2 = MSRInitializer(nn.Linear(InputChannels, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
        
        self.NonLinearity1 = BiasedActivation(InputChannels)
        self.NonLinearity2 = BiasedActivation(LatentDimension)
        
    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        
        x = self.LinearLayer1(self.NonLinearity1(x)).view(x.shape[0], -1)
        return self.NonLinearity2(self.LinearLayer2(x))

class FullyConnectedBlock(nn.Module):
    def __init__(self, LatentDimension):
        super(FullyConnectedBlock, self).__init__()

        self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(LatentDimension)
        self.NonLinearity2 = BiasedActivation(LatentDimension)
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y))
        
        return x + y
           
class MappingBlock(nn.Module):
    def __init__(self, NoiseDimension, LatentDimension, Blocks):
        super(MappingBlock, self).__init__()
        
        self.PrologLayer = MSRInitializer(nn.Linear(NoiseDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)

        self.BlockList = nn.ModuleList([FullyConnectedBlock(LatentDimension) for _ in range(Blocks)])
        
        self.NonLinearity = BiasedActivation(LatentDimension)
        self.EpilogLayer = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
        self.EpilogNonLinearity = BiasedActivation(LatentDimension)
        
    def forward(self, z):
        w = self.PrologLayer(z)
        
        for Block in self.BlockList:
            w = Block(w)
          
        w = self.EpilogLayer(self.NonLinearity(w))
        return self.EpilogNonLinearity(w)
      
def ToRGB(InputChannels, ResidualComponent=False):
    return MSRInitializer(nn.Conv2d(InputChannels, 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0 if ResidualComponent else 1)

class Generator(nn.Module):
    def __init__(self, NoiseDimension=512, LatentDimension=512, LatentMappingDepth=8, StageWidths=[1024, 1024, 1024, 1024, 1024, 1024, 512, 256, 128], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        self.LatentLayer = MappingBlock(NoiseDimension, LatentDimension, LatentMappingDepth // 2 - 1)
        
        self.PrologLayer = GeneratorPrologStage(LatentDimension, StageWidths[0], BlocksPerStage[0], CompressionFactor, ReceptiveField)
        self.AggregateProlog = ToRGB(StageWidths[0])
        
        MainLayers = []
        AggregationLayers = []
        for x in range(len(StageWidths) - 1):
            MainLayers += [GeneratorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x + 1], CompressionFactor, ReceptiveField, ResamplingFilter)]
            AggregationLayers += [ToRGB(StageWidths[x + 1], ResidualComponent=True)]    
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayers = nn.ModuleList(AggregationLayers)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)

    def forward(self, z):
        y, ActivationMaps = self.PrologLayer(self.LatentLayer(z))
        AggregatedOutput = self.AggregateProlog(ActivationMaps)

        for Layer, Aggregate in zip(self.MainLayers, self.AggregationLayers):
            y, ActivationMaps = Layer(y, ActivationMaps)
            AggregatedOutput = self.Resampler(AggregatedOutput) + Aggregate(ActivationMaps)
        
        return AggregatedOutput

class Discriminator(nn.Module):
    def __init__(self, LatentDimension=512, StageWidths=[128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        self.FromRGB = MSRInitializer(nn.Conv2d(3, StageWidths[0], kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        
        MainLayers = []
        for x in range(len(StageWidths) - 1):
            MainLayers += [DiscriminatorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField, ResamplingFilter)]
        self.MainLayers = nn.ModuleList(MainLayers)
        
        self.EpilogLayer = DiscriminatorEpilogStage(StageWidths[-1], LatentDimension, BlocksPerStage[-1], CompressionFactor, ReceptiveField)
        self.CriticLayer = MSRInitializer(nn.Linear(LatentDimension, 1))
        
    def forward(self, x):
        x = self.FromRGB(x)

        for Layer in self.MainLayers:
            x = Layer(x)
        
        x = self.EpilogLayer(x)
        return self.CriticLayer(x).view(x.shape[0])