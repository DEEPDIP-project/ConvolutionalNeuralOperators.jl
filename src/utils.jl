using AbstractFFTs: fft, ifft
using KernelAbstractions
using CUDA
using ChainRulesCore

# This code defines utils for the Convolutional Neural Operators (CNO) architecture.
# It includes also the functionals for the downsampler, upsampler, and activation function.
# They will be fundamental for the CNOLayers defined in CNO.jl

function create_filter(T, grid, cutoff; sigma = 1, filter_type = "sinc", force_cpu = false)
    # TODO extend to multiple dimensions
    N = length(grid)
    N2 = Int(N / 2)
    _kernel = zeros(T, N, N)

    if filter_type == "gaussian"
        k = Int(cutoff)
        center = k / 2
        for i = 1:k
            for j = 1:k
                x = center - i
                y = center - j
                _kernel[i, j] = exp(-(x^2 + y^2) / (2 * sigma^2))
            end
        end
    elseif filter_type == "sinc"
        omega = cutoff * pi
        for x = (-N2+1):1:N2
            for y = (-N2+1):1:N2
                _kernel[x+N2, y+N2] = sinc(x * omega) * sinc(y * omega)
            end
        end
        _kernel = circshift(_kernel, (N2, N2))
    elseif filter_type == "lanczos"
        @warn "You should NOT use lanczos for CNO upsampling because this kernel has a low weight in the orthogonal directions, which is exactly the direction where we create high frequencies with a CNO."
        k = Int(cutoff)
        for i = 1:(2*k+1)
            for j = 1:(2*k+1)
                x = i - (k + 1)
                y = j - (k + 1)
                pos = sqrt(x^2 + y^2)
                _kernel[i, j] = sinc(pos) * sinc(pos / k)
            end
        end
    elseif filter_type == "identity"
        _kernel .= 1
    else
        error("Filter type not recognized")
    end

    # normalize the kernel
    _kernel = _kernel / sum(_kernel)

    # Do the fft of the kernel once
    if CUDA.functional() && !force_cpu
        _kernel = CuArray(_kernel)
    end
    K_f = fft(_kernel, (1,2))

    function apply_fitler(x)
        # Perform circular convolution using FFT (notice I am assuming PBC in both directions)
        if x isa SubArray && parent(x) isa CuArray
            # TODO: This should not be done since collect allocates memory,
            # however to make fft work on GPU I have to do this
            x = CuArray(collect(x))
        end
        X_f = fft(x, (1,2))
        filtered_f = X_f .* K_f
        real(ifft(filtered_f,(1,2)))
    end
end

function downsample_kernel!(mydev, x_filter, N, down_factor::Int)
    backend = mydev["bck"]
    workgroupsize = mydev["workgroupsize"]
    T = mydev["T"]

    downsampled_size = (div(N, down_factor), div(N, down_factor), size(x_filter, 3), size(x_filter, 4))
    if x_filter isa CuArray || (x_filter isa SubArray && parent(x_filter) isa CuArray)
        result = CUDA.zeros(T, downsampled_size)
    else
        result = zeros(T, downsampled_size)
    end
    @kernel inbounds=true function dk!(x_filter, result, down_factor::Int)
        i, j, ch, batch = @index(Global, NTuple)

        # Calculate the corresponding indices in the original array for the spatial dimensions
        original_i = (i - 1) * down_factor + 1
        original_j = (j - 1) * down_factor + 1

        # Assign the downsampled value to the result
        result[i, j, ch, batch] = x_filter[original_i, original_j, ch, batch]
    end

    dk!(backend, workgroupsize)(x_filter, result, down_factor; ndrange=downsampled_size)
    result
end

function ChainRulesCore.rrule(::typeof(downsample_kernel!), mydev, x_filter, N, down_factor::Int)
    
    backend = mydev["bck"]
    workgroupsize = mydev["workgroupsize"]
    downsample_kernel!(mydev, x_filter, N, down_factor)

    function downsample_kernel!_pb(result_bar)

        @kernel function dk_pb!(x_filter_bar, result_bar, down_factor)
            i, j, ch, batch = @index(Global, NTuple)
            original_i = (i - 1) * down_factor + 1
            original_j = (j - 1) * down_factor + 1
            x_filter_bar[original_i, original_j, ch, batch] = result_bar[i, j, ch, batch]
        end

        if x_filter isa CuArray || (x_filter isa SubArray && parent(x_filter) isa CuArray)
            x_filter_bar = CUDA.zeros(eltype(x_filter), size(x_filter))
        else
            x_filter_bar = zeros(eltype(x_filter), size(x_filter))
        end

        dk_pb!(backend,workgroupsize)(x_filter_bar, result_bar, down_factor; ndrange=downsampled_size)
        return NoTangent(), NoTangent(), x_filter_bar, NoTangent(), NoTangent()
    end

    return result, downsample_kernel!_pb
end

function create_CNOdownsampler(
    T::Type,
    D::Int,
    N::Int,
    down_factor::Int,
    cutoff,
    filter_type = "sinc";
    force_cpu = false
)
    grid = collect(0.0:(1.0/(N-1)):1.0)
    filter = create_filter(T, grid, cutoff, filter_type = filter_type, force_cpu = force_cpu)
    # The prefactor is the factor by which the energy is conserved (check 'Convolutional Neural Operators for robust and accurate learning of PDEs')
    prefactor = T(1 / down_factor^D)

    if CUDA.functional() && !force_cpu
        backend = CUDABackend()
        workgroupsize = 256
    else
        backend = CPU()
        workgroupsize = 64
    end
    mydev = Dict("bck" => backend, "workgroupsize" => workgroupsize, "T" => T)

    function CNOdownsampler(x)
        x_filter = filter(x) * prefactor

        downsample_kernel!(mydev, x_filter, N, down_factor)
    end
end



function expand_with_zeros(x, T, up_size, up_factor)
    if x isa CuArray || (x isa SubArray && parent(x) isa CuArray)
        x_up = CUDA.zeros(T, up_size..., size(x)[end-1], size(x)[end])
    else
        x_up = zeros(T, up_size..., size(x)[end-1], size(x)[end])
    end
    idx1 = 1:up_factor:size(x_up)[1]
    idx2 = 1:up_factor:size(x_up)[2]
    CUDA.@allowscalar(copyto!(view(x_up, idx1, idx2, :, :), x) )
    return x_up
end

function ChainRulesCore.rrule(::typeof(expand_with_zeros), x, T, up_size, up_factor)
    if x isa SubArray && parent(x) isa CuArray
        # TODO: This should not be done since collect allocates memory,
        # however to make pullback work on GPU I have to do this
        x = CuArray(collect(x))
    end
    y = expand_with_zeros(x, T, up_size, up_factor)
    function expand_with_zeros_pb(ybar)
        if ybar isa CuArray || (ybar isa SubArray && parent(ybar) isa CuArray)
            xbar = CUDA.zeros(T, size(x))
            idx1 = 1:up_factor:size(ybar)[1]
            idx2 = 1:up_factor:size(ybar)[2]
            CUDA.@allowscalar(copyto!(view(xbar, idx1, idx2, :, :), ybar) )
        else
            xbar = zeros(T, size(x))
            xbar .= ybar[1:up_factor:end, 1:up_factor:end, :, :]
        end
        return NoTangent(), xbar, NoTangent(), NoTangent(), NoTangent()
    end
    return y, expand_with_zeros_pb
end

function create_CNOupsampler(
    T::Type,
    D::Int,
    N::Int,
    up_factor::Int,
    cutoff,
    filter_type = "sinc";
    force_cpu = false
)
    D_up = up_factor * N
    up_size = (D_up for _ = 1:D)
    grid_up = collect(0.0:(1.0/(D_up-1)):1.0)
    filter = create_filter(T, grid_up, cutoff, filter_type = filter_type, force_cpu = force_cpu)

    function CNOupsampler(x)
        # Enhance to the upsampled size
        x_up = expand_with_zeros(x, T, up_size, up_factor)
        # then apply the lowpass filter
        filter(x_up)
    end
end

function create_CNOactivation(
    T::Type,
    D::Int,
    N::Int,
    cutoff;
    activation_function = identity,
    filter_type = "sinc",
    force_cpu = false
)
    # the activation function is applied like this:
    # upsamplex2 -> apply activation -> downsamplex2
    us = create_CNOupsampler(T, D, N, 2, cutoff, filter_type, force_cpu=force_cpu)
    ds = create_CNOdownsampler(T, D, N * 2, 2, cutoff, filter_type, force_cpu=force_cpu)
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
