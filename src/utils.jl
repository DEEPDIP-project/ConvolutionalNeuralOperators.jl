using AbstractFFTs: fft, ifft
using KernelAbstractions
using CUDA
using ChainRulesCore



#function expand_with_zeros(x, T, up_size, up_factor)
#    #TODO : replace this with a kernel using KernelAbstractions
#    if x isa CuArray || (x isa SubArray && parent(x) isa CuArray)
#        x_up = CUDA.zeros(T, up_size..., size(x)[end-1], size(x)[end])
#    else
#        x_up = zeros(T, up_size..., size(x)[end-1], size(x)[end])
#    end
#    idx1 = 1:up_factor:size(x_up)[1]
#    idx2 = 1:up_factor:size(x_up)[2]
#    CUDA.@allowscalar(copyto!(view(x_up, idx1, idx2, :, :), x) )
#    return x_up
#end
#
#function ChainRulesCore.rrule(::typeof(expand_with_zeros), x, T, up_size, up_factor)
#    if x isa SubArray && parent(x) isa CuArray
#        # TODO: This should not be done since collect allocates memory,
#        # however to make pullback work on GPU I have to do this
#        x = CuArray(collect(x))
#    end
#    y = expand_with_zeros(x, T, up_size, up_factor)
#    function expand_with_zeros_pb(ybar)
#        if ybar isa CuArray || (ybar isa SubArray && parent(ybar) isa CuArray)
#            xbar = CUDA.zeros(T, size(x))
#            idx1 = 1:up_factor:size(ybar)[1]
#            idx2 = 1:up_factor:size(ybar)[2]
#            CUDA.@allowscalar(copyto!(view(xbar, idx1, idx2, :, :), ybar) )
#        else
#            xbar = zeros(T, size(x))
#            xbar .= ybar[1:up_factor:end, 1:up_factor:end, :, :]
#        end
#        return NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
#    end
#    return y, expand_with_zeros_pb
#end


function create_CNOactivation(
    T::Type,
    D::Int,
    N::Int,
    cutoff;
    activation_function = identity,
    filter_type = "sinc",
    force_cpu = false,
)
    # the activation function is applied like this:
    # upsamplex2 -> apply activation -> downsamplex2
    us = create_CNOupsampler(T, D, N, 2, cutoff, filter_type, force_cpu = force_cpu)
    ds = create_CNOdownsampler(T, D, N * 2, 2, cutoff, filter_type, force_cpu = force_cpu)
    function CNOactivation(x)
        ds(activation_function(us(x)))
    end
end

function ch_to_ranges(arr::Vector{Int})
    # TODO write the docstring for this
    # it returns a vector containing range_k
    ranges = Vector{UnitRange{Int}}()
    start = 1
    for n in arr
        push!(ranges, start:(start+n-1))
        start += n
    end
    return ranges
end

function ch_to_bottleneck_ranges(bot_d::Vector{Int}, ch_size::Vector{Int})
    # TODO write the docstring for this
    # it returns a vector containing tuple of (range_k, bottleneck_extra_info)
    @assert length(bot_d) == length(ch_size) "The bottleneck depth and the channel size must have the same length"
    ranges = Vector{Any}()
    start = 1
    for (i, nblock) in enumerate(bot_d)
        this_bottleneck = ()
        # these are all the resblocks
        for x = 1:(2*(nblock-1))
            this_bottleneck = (this_bottleneck..., (start:(start+ch_size[i]-1),))
            start = start + ch_size[i]
        end
        # then this is the last block
        this_bottleneck = (this_bottleneck..., (start:(start+ch_size[i]-1),))
        push!(ranges, this_bottleneck)
        start += ch_size[i]
    end
    return ranges
end
