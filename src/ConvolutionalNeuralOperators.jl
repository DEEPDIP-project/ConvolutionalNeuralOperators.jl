module ConvolutionalNeuralOperators

using CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array
using KernelAbstractions
using Lux
using LuxCUDA
using ChainRulesCore
using AbstractFFTs: fft, ifft
using Random: AbstractRNG

include("filters.jl")
include("convolution.jl")
include("downsample.jl")
include("upsample.jl")
include("utils.jl")
include("models.jl")

export create_CNO, create_CNOdownsampler, create_CNOupsampler, create_CNOactivation, cno

end
