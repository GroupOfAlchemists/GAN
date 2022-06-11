import math
import numpy
import torch
import torch.nn as nn

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()
    
    return Layer

class BiasedActivation(nn.Module):
    Gain = math.sqrt(2)
    Function = nn.functional.silu
    
    def __init__(self, InputUnits, ConvolutionalLayer=True):
        super(BiasedActivation, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
        self.ConvolutionalLayer = ConvolutionalLayer
        
    def forward(self, x):
        y = x + self.Bias.view(1, -1, 1, 1) if self.ConvolutionalLayer else x + self.Bias.view(1, -1)
        return BiasedActivation.Function(y)

class GeneratorBlock(nn.Module):
      def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
          super(GeneratorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(InputChannels)
          
      def forward(self, x, ActivationMaps):
          y = self.LinearLayer1(ActivationMaps)
          y = self.LinearLayer2(self.NonLinearity1(y))
          
          y = x + y
          
          return y, self.NonLinearity2(y)

class DiscriminatorBlock(nn.Module):
      def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
          super(DiscriminatorBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, InputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x))
          y = self.LinearLayer2(self.NonLinearity2(y))
          
          return x + y

def CreateLowpassKernel(Weights):
    Kernel = numpy.array([Weights])
    Kernel = torch.Tensor(Kernel.T @ Kernel)
    Kernel = Kernel / torch.sum(Kernel)
    return Kernel.view(1, 1, Kernel.shape[0], Kernel.shape[1])

class InplaceBilinearUpsampler(nn.Module):
      def __init__(self):
          super(InplaceBilinearUpsampler, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel([1., 2., 1.]))
          
      def forward(self, x):
          x = nn.functional.pixel_shuffle(x, 2)
          y = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
          
          return nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=1).view(*x.shape)
          
class InplaceBilinearDownsampler(nn.Module):
      def __init__(self):
          super(InplaceBilinearDownsampler, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel([1., 2., 1.]))
          
      def forward(self, x):
          y = nn.functional.pad(x, (1, 1, 1, 1), mode='reflect')
          y = nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=1).view(*x.shape)

          return nn.functional.pixel_unshuffle(y, 2)

class GeneratorUpsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField):
          super(GeneratorUpsampleBlock, self).__init__()
          
          CompressedChannels = InputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels, CompressedChannels * 4, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(CompressedChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          self.NonLinearity3 = BiasedActivation(OutputChannels)
          
          self.Resampler = InplaceBilinearUpsampler()
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))

      def forward(self, x, ActivationMaps):
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)
          
          y = self.LinearLayer1(ActivationMaps)
          y = self.Resampler(self.LinearLayer2(self.NonLinearity1(y)))
          y = self.LinearLayer3(self.NonLinearity2(y))
          
          y = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, antialias=False) + y
          
          return y, self.NonLinearity3(y)

class DiscriminatorDownsampleBlock(nn.Module):
      def __init__(self, InputChannels, OutputChannels, CompressionFactor, ReceptiveField):
          super(DiscriminatorDownsampleBlock, self).__init__()
          
          CompressedChannels = OutputChannels // CompressionFactor
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, CompressedChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer2 = MSRInitializer(nn.Conv2d(CompressedChannels * 4, CompressedChannels, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=BiasedActivation.Gain)
          self.LinearLayer3 = MSRInitializer(nn.Conv2d(CompressedChannels, OutputChannels, kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=0)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(CompressedChannels)
          self.NonLinearity3 = BiasedActivation(CompressedChannels)
          
          self.Resampler = InplaceBilinearDownsampler()
          if InputChannels != OutputChannels:
              self.ShortcutLayer = MSRInitializer(nn.Conv2d(InputChannels, OutputChannels, kernel_size=1, stride=1, padding=0, bias=False))
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x))
          y = self.LinearLayer2(self.Resampler(self.NonLinearity2(y)))
          y = self.LinearLayer3(self.NonLinearity3(y))
          
          x = nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, antialias=True, recompute_scale_factor=True)
          if hasattr(self, 'ShortcutLayer'):
              x = self.ShortcutLayer(x)

          return x + y
     
class GeneratorStage(nn.Module):
      def __init__(self, InputChannels, OutputChannels, Blocks, CompressionFactor, ReceptiveField):
          super(GeneratorStage, self).__init__()
          
          self.BlockList = nn.ModuleList([GeneratorBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)] + [GeneratorUpsampleBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField)])
          
      def forward(self, x, ActivationMaps):
          for Block in self.BlockList:
              x, ActivationMaps = Block(x, ActivationMaps)
          return x, ActivationMaps
          
class DiscriminatorStage(nn.Module):
      def __init__(self, InputChannels, OutputChannels, Blocks, CompressionFactor, ReceptiveField):
          super(DiscriminatorStage, self).__init__()

          self.BlockList = nn.ModuleList([DiscriminatorDownsampleBlock(InputChannels, OutputChannels, CompressionFactor, ReceptiveField)] + [DiscriminatorBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(Blocks - 1)])
        
      def forward(self, x):
          for Block in self.BlockList:
              x = Block(x)
          return x
        
class GeneratorPrologLayer(nn.Module):
    def __init__(self, LatentDimension, OutputChannels):
        super(GeneratorPrologLayer, self).__init__()
        
        self.Basis = nn.Parameter(torch.empty((OutputChannels, 4, 4)))
        self.LinearLayer = MSRInitializer(nn.Linear(LatentDimension, OutputChannels, bias=False))
        self.NonLinearity = BiasedActivation(OutputChannels)
        
        self.Basis.data.normal_(0, BiasedActivation.Gain)
        
    def forward(self, w):
        x = self.LinearLayer(w).view(w.shape[0], -1, 1, 1)
        y = self.Basis.view(1, -1, 4, 4) * x
        return y, self.NonLinearity(y)
     
class DiscriminatorEpilogLayer(nn.Module):
      def __init__(self, InputChannels, LatentDimension):
          super(DiscriminatorEpilogLayer, self).__init__()
          
          self.LinearLayer1 = MSRInitializer(nn.Conv2d(InputChannels, InputChannels, kernel_size=4, stride=1, padding=0, groups=InputChannels, bias=False))
          self.LinearLayer2 = MSRInitializer(nn.Linear(InputChannels, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
          
          self.NonLinearity1 = BiasedActivation(InputChannels)
          self.NonLinearity2 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          
      def forward(self, x):
          y = self.LinearLayer1(self.NonLinearity1(x)).view(x.shape[0], -1)
          return self.NonLinearity2(self.LinearLayer2(y))

class FullyConnectedBlock(nn.Module):
    def __init__(self, LatentDimension):
        super(FullyConnectedBlock, self).__init__()

        self.LinearLayer1 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
        self.LinearLayer2 = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=0)
        
        self.NonLinearity1 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
        self.NonLinearity2 = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
        
    def forward(self, x):
        y = self.LinearLayer1(self.NonLinearity1(x))
        y = self.LinearLayer2(self.NonLinearity2(y))
        
        return x + y
           
class MappingBlock(nn.Module):
      def __init__(self, NoiseDimension, LatentDimension, Blocks):
          super(MappingBlock, self).__init__()
          
          self.PrologLayer = MSRInitializer(nn.Linear(NoiseDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)

          self.BlockList = nn.ModuleList([FullyConnectedBlock(LatentDimension) for _ in range(Blocks)])
          
          self.NonLinearity = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          self.EpilogLayer = MSRInitializer(nn.Linear(LatentDimension, LatentDimension, bias=False), ActivationGain=BiasedActivation.Gain)
          self.EpilogNonLinearity = BiasedActivation(LatentDimension, ConvolutionalLayer=False)
          
      def forward(self, z):
          w = self.PrologLayer(z)
          
          for Block in self.BlockList:
              w = Block(w)
              
          w = self.EpilogLayer(self.NonLinearity(w))
          return self.EpilogNonLinearity(w)
      
def ToRGB(InputChannels, ResidualComponent=False):
    return MSRInitializer(nn.Conv2d(InputChannels, 3, kernel_size=1, stride=1, padding=0, bias=False), ActivationGain=0 if ResidualComponent else 1)

class Generator(nn.Module):
    def __init__(self, NoiseDimension=512, LatentDimension=512, LatentMappingDepth=8, PrologWidth=1024, StageWidths=[1024, 512, 512, 512, 256, 256, 256, 128], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3):
        super(Generator, self).__init__()
        
        self.LatentLayer = MappingBlock(NoiseDimension, LatentDimension, LatentMappingDepth // 2 - 1)
        
        self.PrologLayer = GeneratorPrologLayer(LatentDimension, PrologWidth)
        self.AggregateProlog = ToRGB(PrologWidth)
        
        MainLayers = []
        AggregationLayers = []
        ExtendedStageWidths = [PrologWidth] + StageWidths
        for x in range(len(StageWidths)):
            MainLayers += [GeneratorStage(ExtendedStageWidths[x], ExtendedStageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField)]
            AggregationLayers += [ToRGB(ExtendedStageWidths[x + 1], ResidualComponent=True)]    
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayers = nn.ModuleList(AggregationLayers)

    def forward(self, z, EnableLatentMapping=True):
        w = self.LatentLayer(z) if EnableLatentMapping else z
        
        y, ActivationMaps = self.PrologLayer(w)
        AggregatedOutput = self.AggregateProlog(ActivationMaps)

        for Layer, Aggregate in zip(self.MainLayers, self.AggregationLayers):
            y, ActivationMaps = Layer(y, ActivationMaps)
            AggregatedOutput = nn.functional.interpolate(AggregatedOutput, scale_factor=2, mode='bilinear', align_corners=False, antialias=False) + Aggregate(ActivationMaps)
        
        return AggregatedOutput

class Discriminator(nn.Module):
    def __init__(self, LatentDimension=512, EpilogWidth=1024, StageWidths=[128, 256, 256, 256, 512, 512, 512, 1024], BlocksPerStage=[4, 4, 4, 4, 4, 4, 4, 4], CompressionFactor=4, ReceptiveField=3):
        super(Discriminator, self).__init__()
        
        self.FromRGB = MSRInitializer(nn.Conv2d(3, StageWidths[0], kernel_size=ReceptiveField, stride=1, padding=(ReceptiveField - 1) // 2, padding_mode='reflect', bias=False), ActivationGain=BiasedActivation.Gain)
        
        MainLayers = []
        ExtendedStageWidths = StageWidths + [EpilogWidth]
        for x in range(len(StageWidths)):
            MainLayers += [DiscriminatorStage(ExtendedStageWidths[x], ExtendedStageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField)]
        self.MainLayers = nn.ModuleList(MainLayers)
        
        self.EpilogLayer = DiscriminatorEpilogLayer(EpilogWidth, LatentDimension)
        self.CriticLayer = MSRInitializer(nn.Linear(LatentDimension, 1))
        
    def forward(self, x):
        x = self.FromRGB(x)

        for Layer in self.MainLayers:
            x = Layer(x)
        
        x = self.EpilogLayer(x)
        return self.CriticLayer(x).view(x.shape[0])