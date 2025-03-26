
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
