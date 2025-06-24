# src/models/quantization.py

import torch
import torch.nn as nn

class FakeQuantizeSTEFunction(torch.autograd.Function):
    """
    Implements Straight-Through Estimator for symmetric quantization.
    Forward: quantizes to qmin/qmax range and dequantizes.
    Backward: passes gradient as is.
    """
    @staticmethod
    def forward(ctx, x, scale, qmin, qmax):
        # Symmetric quantization: zero_point is 0
        # x_quant = torch.round(x / scale)
        # xq_clamped = torch.clamp(x_quant, qmin, qmax)
        # xdq = xq_clamped * scale
        
        # Simplified and common STE approach for q-dq
        # This ensures that the values are on the quantization grid
        xq_clamped = torch.clamp(torch.round(x / scale), qmin, qmax)
        xdq = xq_clamped * scale
        return xdq

    @staticmethod
    def backward(ctx, grad_output):
        # Pass the gradient through unmodified (Straight-Through Estimator)
        return grad_output, None, None, None


class FakeQuantizeRepresentation(nn.Module):
    """
    Fake Quantization module for representations.
    Applies symmetric quantization and dequantization during training to simulate noise,
    using Straight-Through Estimator for gradients.
    """
    def __init__(self, bits=8):
        super().__init__()
        self.bits = bits
        if bits == 8:
            self.qmin = -128
            self.qmax = 127
        else:
            # Placeholder for other bitwidths, symmetric
            self.qmin = -2**(self.bits - 1)
            self.qmax = 2**(self.bits - 1) - 1

    def forward(self, x):
        # Per-tensor symmetric quantization
        # Calculate scale based on the maximum absolute value in the tensor (detached for observer)
        abs_max = torch.max(torch.abs(x.detach()))
        scale = abs_max / self.qmax
        
        # Ensure a minimum scale to prevent division by zero if x is all zeros or very small
        scale = torch.clamp(scale, min=1e-8)

        # Apply fake quantization using the STE function
        quantized_dequantized_x = FakeQuantizeSTEFunction.apply(x, scale, self.qmin, self.qmax)
        return quantized_dequantized_x


# Helper functions for actual quantization (used in validation/inference)
def quantize_tensor_symmetric(x_f32, bits=8):
    """
    Actually quantizes an f32 tensor to int representation (e.g., int8)
    using symmetric quantization.
    Returns the quantized integer tensor and the scale factor.
    """
    if bits == 8:
        qmin = -128
        qmax = 127
        dtype = torch.int8
    else:
        qmin = -2**(bits - 1)
        qmax = 2**(bits - 1) - 1
        if bits <= 8:
            dtype = torch.int8
        elif bits <= 16:
            dtype = torch.int16
        else: # up to 32
            dtype = torch.int32


    abs_max = torch.max(torch.abs(x_f32.detach()))
    scale = abs_max / qmax
    scale = torch.clamp(scale, min=1e-8)
    
    x_quant = torch.round(x_f32 / scale)
    x_quant_clamped = torch.clamp(x_quant, qmin, qmax).to(dtype)
    return x_quant_clamped, scale

def dequantize_tensor_symmetric(x_quantized, scale):
    """
    Dequantizes an integer tensor back to f32 using the scale factor.
    """
    return x_quantized.float() * scale