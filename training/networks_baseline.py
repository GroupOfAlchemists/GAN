import numpy as np
import torch
import torch.nn as nn
from torch_utils.ops import upfirdn2d
from torch_utils.ops import conv2d_resample
from torch_utils.ops import bias_act
import math

def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0,  ActivationGain / math.sqrt(FanIn))
    return Layer


class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        init_gain       = 1,
        padding         = None,
        groups          = 1,
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1]     # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2 if padding is None else padding
        self.groups = groups

        memory_format = torch.contiguous_format
        self.weight = torch.nn.Parameter(torch.empty([out_channels, in_channels // groups, kernel_size, kernel_size]).to(memory_format=memory_format))
        MSRInitializer(self, init_gain)

    def forward(self, x):
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=self.weight.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, groups=self.groups, flip_weight=flip_weight)
        
        return x
    
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features               # Number of output features.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]))
        MSRInitializer(self)

    def forward(self, x):
        x = x.matmul(self.weight.to(x.dtype).t())
        return x

class BiasedActivation(nn.Module):
    act_gain = bias_act.activation_funcs['lrelu'].def_gain
    
    def __init__(self, InputUnits):
        super(BiasedActivation, self).__init__()
        
        self.Bias = nn.Parameter(torch.empty(InputUnits))
        self.Bias.data.zero_()
        
    def forward(self, x):
        return bias_act.bias_act(x, self.Bias, act='lrelu', gain=1)
    
class ResidualBlock(nn.Module):
    def __init__(self, InputChannels, CompressionFactor, ReceptiveField):
        super(ResidualBlock, self).__init__()
        
        CompressedChannels = InputChannels // CompressionFactor
        
        self.LinearLayer1 = Conv2dLayer(InputChannels, CompressedChannels, kernel_size=1, init_gain=BiasedActivation.act_gain)
        self.LinearLayer2 = Conv2dLayer(CompressedChannels, CompressedChannels, kernel_size=ReceptiveField, init_gain=BiasedActivation.act_gain)
        self.LinearLayer3 = Conv2dLayer(CompressedChannels, InputChannels, kernel_size=1, init_gain=0)
        
        self.NonLinearity1 = BiasedActivation(CompressedChannels)
        self.NonLinearity2 = BiasedActivation(CompressedChannels)
        
    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))
        
        return x + y
    
class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Conv2dLayer(InputChannels, OutputChannels, kernel_size=1, up=2, resample_filter=ResamplingFilter)
        else:
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(ResamplingFilter))
        
    def forward(self, x):
        if hasattr(self, 'LinearLayer'):
            return self.LinearLayer(x)
        else:
            return upfirdn2d.upsample2d(x, self.resample_filter)
        
class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()
        
        if InputChannels != OutputChannels:
            self.LinearLayer = Conv2dLayer(InputChannels, OutputChannels, kernel_size=1, down=2, resample_filter=ResamplingFilter)
        else:
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(ResamplingFilter))
        
    def forward(self, x):
        if hasattr(self, 'LinearLayer'):
            return self.LinearLayer(x)
        else:
            return upfirdn2d.downsample2d(x, self.resample_filter)
        
class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels):
        super(GenerativeBasis, self).__init__()
        
        self.Basis = torch.nn.Parameter(torch.randn([OutputChannels, 4, 4]))
        self.LinearLayer = FullyConnectedLayer(InputDimension, OutputChannels)
        
    def forward(self, x):
        return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(x.shape[0], -1, 1, 1)
    
class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()
        
        self.Basis = Conv2dLayer(InputChannels, InputChannels, kernel_size=4, padding=0, groups=InputChannels)
        self.LinearLayer = FullyConnectedLayer(InputChannels, OutputDimension)
        
    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))
    
class GeneratorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, CompressionFactor, ReceptiveField, ResamplingFilter=None):
        super(GeneratorStage, self).__init__()
        
        TransitionLayer = GenerativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([TransitionLayer] + [ResidualBlock(OutputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks)])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class DiscriminatorStage(nn.Module):
    def __init__(self, InputChannels, OutputChannels, NumberOfBlocks, CompressionFactor, ReceptiveField, ResamplingFilter=None):
        super(DiscriminatorStage, self).__init__()
        
        TransitionLayer = DiscriminativeBasis(InputChannels, OutputChannels) if ResamplingFilter is None else DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        self.Layers = nn.ModuleList([ResidualBlock(InputChannels, CompressionFactor, ReceptiveField) for _ in range(NumberOfBlocks)] + [TransitionLayer])
        
    def forward(self, x):
        for Layer in self.Layers:
            x = Layer(x)
        
        return x
    
class Generator(nn.Module):
    def __init__(self, NoiseDimension, StageWidths, BlocksPerStage, CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1,3,3,1]):
        super(Generator, self).__init__()
        
        MainLayers = [GeneratorStage(NoiseDimension, StageWidths[0], BlocksPerStage[0], CompressionFactor, ReceptiveField)]
        MainLayers += [GeneratorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x + 1], CompressionFactor, ReceptiveField, ResamplingFilter) for x in range(len(StageWidths) - 1)]
        
        AggregationLayers = [Conv2dLayer(StageWidths[0], 3, kernel_size=1)]
        AggregationLayers += [Conv2dLayer(StageWidths[x + 1], 3, kernel_size=1, init_gain=0) for x in range(len(StageWidths) - 1)]
        
        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayers = nn.ModuleList(AggregationLayers)
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(ResamplingFilter))
        
        self.z_dim = NoiseDimension
        
    def forward(self, x):
        AggregatedOutput = None
        
        for Layer, Aggregate in zip(self.MainLayers, self.AggregationLayers):
            x = Layer(x)
            AggregatedOutput = upfirdn2d.upsample2d(AggregatedOutput, self.resample_filter) + Aggregate(x) if AggregatedOutput is not None else Aggregate(x)
        
        return AggregatedOutput
    
class Discriminator(nn.Module):
    def __init__(self, StageWidths, BlocksPerStage, CompressionFactor=4, ReceptiveField=3, ResamplingFilter=[1,3,3,1]):
        super(Discriminator, self).__init__()
        
        MainLayers = [DiscriminatorStage(StageWidths[x], StageWidths[x + 1], BlocksPerStage[x], CompressionFactor, ReceptiveField, ResamplingFilter) for x in range(len(StageWidths) - 1)]
        MainLayers += [DiscriminatorStage(StageWidths[-1], 1, BlocksPerStage[-1], CompressionFactor, ReceptiveField)]
        
        self.ExtractionLayer = Conv2dLayer(3, StageWidths[0], kernel_size=ReceptiveField)
        self.MainLayers = nn.ModuleList(MainLayers)
        
    def forward(self, x):
        x = self.ExtractionLayer(x)
        
        for Layer in self.MainLayers:
            x = Layer(x)
        
        return x.view(x.shape[0])