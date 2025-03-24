module ConvolutionalNeuralOperators

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("filters.jl")
include("convolution.jl")
include("downsample.jl")
include("upsample.jl")
include("utils.jl")
include("models.jl")

export create_CNO, create_CNOdownsampler, create_CNOupsampler, create_CNOactivation, cno

end
