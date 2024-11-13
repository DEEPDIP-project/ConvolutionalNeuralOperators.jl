module ConvolutionalNeuralOperators

using CUDA: CUDA
ArrayType = CUDA.functional() ? CUDA.CuArray : Array

include("utils.jl")
include("models.jl")

export create_CNO, create_CNOdownsampler, create_CNOupsampler, create_CNOactivation

end
