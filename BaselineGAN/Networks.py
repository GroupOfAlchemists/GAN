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

class GeneralBlock(nn.Module):
    def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
        super(GeneralBlock, self).__init__()
        
        CompressedChannels = InputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedChannels)
        self.NonLinearity2 = BiasedActivation(CompressedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
class UpsampleBlock(nn.Module):
    def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter):
        super(UpsampleBlock, self).__init__()
        
        CompressedInputChannels = InputChannels // CompressionFactor
        CompressedOutputChannels = OutputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedInputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedInputChannels, CompressedOutputChannels * 4, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedOutputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedInputChannels)
        self.NonLinearity2 = BiasedActivation(CompressedOutputChannels)
        
        self.MainResampler = InplaceUpsampler(ResamplingFilter)
        self.ShortcutResampler = InterpolativeUpsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        Identity = self.ShortcutLayer(x) if hasattr(self, 'ShortcutLayer') else x
        Identity = self.ShortcutResampler(Identity)
        
        y = self.LinearLayer1(x)
        y = self.MainResampler(self.LinearLayer2(self.NonLinearity1(y)))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return Identity + y
    
class DownsampleBlock(nn.Module):
    def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter):
        super(DownsampleBlock, self).__init__()
        
        CompressedInputChannels = InputChannels // CompressionFactor
        CompressedOutputChannels = OutputChannels // CompressionFactor
        
        self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedInputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedInputChannels * 4, CompressedOutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedOutputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedInputChannels)
        self.NonLinearity2 = BiasedActivation(CompressedOutputChannels)
        
        self.MainResampler = InplaceDownsampler(ResamplingFilter)
        self.ShortcutResampler = InterpolativeDownsampler(ResamplingFilter)
        
        if InputChannels != OutputChannels:
            self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
        
    def forward(self, x):
        Identity = self.ShortcutResampler(x)
        Identity = self.ShortcutLayer(Identity) if hasattr(self, 'ShortcutLayer') else Identity
        
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.MainResampler(self.NonLinearity1(y)))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return Identity + y
    
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, CompressionFactor, ReceptiveField, TransitionBlock=None, ResamplingFilter=None):
        super(GeneratorStage, self).__init__()
        
        if TransitionBlock is not None:
            self.BlockList = nn.ModuleList([TransitionBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter)] + [GeneralBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks - 1)])
        else:
            assert InputChannels == OutputChannels
            self.BlockList = nn.ModuleList([GeneralBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        
        return x
    
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, CompressionFactor, ReceptiveField, TransitionBlock=None, ResamplingFilter=None):
        super(DiscriminatorStage, self).__init__()
        
        if TransitionBlock is not None:
            self.BlockList = nn.ModuleList([GeneralBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks - 1)] + [TransitionBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField, ResamplingFilter)])
        else:
            assert InputChannels == OutputChannels
            self.BlockList = nn.ModuleList([GeneralBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks)])
      
    def forward(self, x):
        for Block in self.BlockList:
            x = Block(x)
        
        return x
    
class PrologLayer(nn.Module):
    def __init__(self, InputUnits, OutputChannels):
        super(PrologLayer, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Linear(InputUnits, OutputChannels, bias=False))
        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        
    def forward(self, x):
        return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(x.shape[0], -1, 1, 1)
    
class EpilogLayer(nn.Module):
    def __init__(self, InputChannels):
        super(EpilogLayer, self).__init__()
        
        self.LinearLayer = MSRInitializer(nn.Linear(InputChannels, 1, bias=False))
        self.Basis = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1)).view(x.shape[0])
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension=512, StageWidths=[1024, 1024, 1024, 1024, 1024, 1024, 512, 256, 128], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Generator, self).__init__()
        
        MainLayers = [GeneratorStage(StageWidths[0], StageWidths[0], BlocksPerStage[0], CompressionFactor, ReceptiveField)]
        MainLayers += [GeneratorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x + 1], CompressionFactor, ReceptiveField, UpsampleBlock, ResamplingFilter) for x in range(len(StageWidths) - 1)]
        
        AggregationLayers = [MSRInitializer(nn.Conv2d(StageWidths[0], 3, kernel_size=1, stride=1, padding=0, bias=False))]
        AggregationLayers += [MSRInitializer(nn.Conv2d(StageWidths[x + 1], 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0) for x in range(len(StageWidths) - 1)]
        
        self.FeatureLayer = PrologLayer(NoiseDimension, StageWidths[0])
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayers = nn.ModuleList(AggregationLayers)
        self.Resampler = InterpolativeUpsampler(ResamplingFilter)
        
    def forward(self, z):
        x = self.MainLayers[0](self.FeatureLayer(z))
        AggregatedOutput = self.AggregationLayers[0](x)
        
        for Layer, Aggregate in zip(self.MainLayers[1:], self.AggregationLayers[1:]):
            x = Layer(x)
            AggregatedOutput = self.Resampler(AggregatedOutput) + Aggregate(x)
        
        return AggregatedOutput
    
class Discriminator(nn.Module):
    def __init__(self, StageWidths=[128, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1, 2, 1]):
        super(Discriminator, self).__init__()
        
        MainLayers = [DiscriminatorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField, DownsampleBlock, ResamplingFilter) for x in range(len(StageWidths) - 1)]
        MainLayers += [DiscriminatorStage(StageWidths[-1], StageWidths[-1], BlocksPerStage[-1], CompressionFactor, ReceptiveField)]
        
        self.ExtractionLayer = MSRInitializer(nn.Conv2d(3, StageWidths[0], kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False))
        self.MainLayers = nn.ModuleList(MainLayers)
        self.CriticLayer = EpilogLayer(StageWidths[-1])
        
    def forward(self, x):
        x = self.ExtractionLayer(x)
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        return self.CriticLayer(x)