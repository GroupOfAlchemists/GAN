import torch
import torch.nn as nn
import numpy
from torch_utils.ops import upfirdn2d

def CreateLowpassKernel(Weights, Inplace):
    Kernel = numpy.array([Weights]) if Inplace else numpy.convolve(Weights, [1, 1]).reshape(1, -1)
    Kernel = torch.Tensor(Kernel.T @ Kernel)
    return Kernel / torch.sum(Kernel)

class InterpolativeUpsamplerReference(nn.Module):
      def __init__(self, Filter):
          super(InterpolativeUpsamplerReference, self).__init__()
          
          Kernel = 4 * CreateLowpassKernel(Filter, Inplace=False)
          self.register_buffer('Kernel', Kernel.view(1, 1, Kernel.shape[0], Kernel.shape[1]))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          y = nn.functional.conv_transpose2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=2, padding=3 * self.FilterRadius)
          
          return y.view(x.shape[0], x.shape[1], y.shape[2], y.shape[3])
      
class InterpolativeDownsamplerReference(nn.Module):
      def __init__(self, Filter):
          super(InterpolativeDownsamplerReference, self).__init__()
          
          Kernel = CreateLowpassKernel(Filter, Inplace=False)
          self.register_buffer('Kernel', Kernel.view(1, 1, Kernel.shape[0], Kernel.shape[1]))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          y = nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=2)
          
          return y.view(x.shape[0], x.shape[1], y.shape[2], y.shape[3])
      
class InplaceUpsamplerReference(nn.Module):
      def __init__(self, Filter):
          super(InplaceUpsamplerReference, self).__init__()
          
          Kernel = CreateLowpassKernel(Filter, Inplace=True)
          self.register_buffer('Kernel', Kernel.view(1, 1, Kernel.shape[0], Kernel.shape[1]))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          x = nn.functional.pixel_shuffle(x, 2)
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          
          return nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=1).view(*x.shape)
      
class InplaceDownsamplerReference(nn.Module):
      def __init__(self, Filter):
          super(InplaceDownsamplerReference, self).__init__()
          
          Kernel = CreateLowpassKernel(Filter, Inplace=True)
          self.register_buffer('Kernel', Kernel.view(1, 1, Kernel.shape[0], Kernel.shape[1]))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          y = nn.functional.conv2d(y.view(y.shape[0] * y.shape[1], 1, y.shape[2], y.shape[3]), self.Kernel, stride=1).view(*x.shape)

          return nn.functional.pixel_unshuffle(y, 2)

class InterpolativeUpsamplerCUDA(nn.Module):
      def __init__(self, Filter):
          super(InterpolativeUpsamplerCUDA, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          return upfirdn2d.upsample2d(y, self.Kernel, padding=-2 * self.FilterRadius)

class InterpolativeDownsamplerCUDA(nn.Module):
      def __init__(self, Filter):
          super(InterpolativeDownsamplerCUDA, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=False))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          return upfirdn2d.downsample2d(y, self.Kernel, padding=-self.FilterRadius)

class InplaceUpsamplerCUDA(nn.Module):
      def __init__(self, Filter):
          super(InplaceUpsamplerCUDA, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          x = nn.functional.pixel_shuffle(x, 2)
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          
          return upfirdn2d.upfirdn2d(y, self.Kernel)

class InplaceDownsamplerCUDA(nn.Module):
      def __init__(self, Filter):
          super(InplaceDownsamplerCUDA, self).__init__()
          
          self.register_buffer('Kernel', CreateLowpassKernel(Filter, Inplace=True))
          self.FilterRadius = len(Filter) // 2
          
      def forward(self, x):
          y = nn.functional.pad(x, (self.FilterRadius, self.FilterRadius, self.FilterRadius, self.FilterRadius), mode='reflect')
          y = upfirdn2d.upfirdn2d(y, self.Kernel)

          return nn.functional.pixel_unshuffle(y, 2)

InterpolativeUpsampler = InterpolativeUpsamplerCUDA
InterpolativeDownsampler = InterpolativeDownsamplerCUDA
InplaceUpsampler = InplaceUpsamplerCUDA
InplaceDownsampler = InplaceDownsamplerCUDA