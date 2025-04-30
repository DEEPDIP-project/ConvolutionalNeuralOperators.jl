function downsample_kernel(mydev, x_filter, N, down_factor::Int)
    backend = mydev["bck"]
    workgroupsize = mydev["workgroupsize"]
    T = mydev["T"]

    downsampled_size =
        (div(N, down_factor), div(N, down_factor), size(x_filter, 3), size(x_filter, 4))
    if x_filter isa CuArray || (x_filter isa SubArray && parent(x_filter) isa CuArray)
        result = CUDA.zeros(T, downsampled_size)
    else
        result = zeros(T, downsampled_size)
    end
    @kernel inbounds = true function dk!(x_filter, result, down_factor::Int)
        i, j, ch, batch = @index(Global, NTuple)

        # Calculate the corresponding indices in the original array for the spatial dimensions
        original_i = (i - 1) * down_factor + 1
        original_j = (j - 1) * down_factor + 1

        # Assign the downsampled value to the result
        result[i, j, ch, batch] = x_filter[original_i, original_j, ch, batch]
    end

    dk!(backend, workgroupsize)(x_filter, result, down_factor; ndrange = downsampled_size)
    result
end

function ChainRulesCore.rrule(
    ::typeof(downsample_kernel),
    mydev,
    x_filter,
    N,
    down_factor::Int,
)

    backend = mydev["bck"]
    workgroupsize = mydev["workgroupsize"]
    downsampled_size =
        (div(N, down_factor), div(N, down_factor), size(x_filter, 3), size(x_filter, 4))
    result = downsample_kernel(mydev, x_filter, N, down_factor)

    function downsample_kernel_pb(result_bar)

        if x_filter isa CuArray || (x_filter isa SubArray && parent(x_filter) isa CuArray)
            x_filter_bar = CUDA.zeros(eltype(x_filter), size(x_filter))
        else
            x_filter_bar = zeros(eltype(x_filter), size(x_filter))
        end

        @kernel inbounds = true function dk_pb!(x_filter_bar, result_bar, down_factor)
            i, j, ch, batch = @index(Global, NTuple)
            original_i = (i - 1) * down_factor + 1
            original_j = (j - 1) * down_factor + 1
            x_filter_bar[original_i, original_j, ch, batch] = result_bar[i, j, ch, batch]
        end

        dk_pb!(backend, workgroupsize)(
            x_filter_bar,
            result_bar,
            down_factor;
            ndrange = downsampled_size,
        )

        return NoTangent(), NoTangent(), x_filter_bar, NoTangent(), NoTangent()
    end

    return result, downsample_kernel_pb
end

function create_CNOdownsampler(
    T::Type,
    D::Int,
    N::Int,
    down_factor::Int,
    cutoff,
    filter_type = "sinc";
    force_cpu = false,
)
    grid = collect(0.0:(1.0/(N-1)):1.0)
    filter =
        create_filter(T, grid, cutoff, filter_type = filter_type, force_cpu = force_cpu)
    # The prefactor is the factor by which the energy is conserved (check 'Convolutional Neural Operators for robust and accurate learning of PDEs')
    prefactor = T(1 / down_factor^D)

    if CUDA.functional() && !force_cpu
        backend = CUDABackend()
        workgroupsize = 256
    else
        backend = KernelAbstractions.CPU()
        workgroupsize = 64
    end
    mydev = Dict("bck" => backend, "workgroupsize" => workgroupsize, "T" => T)

    function CNOdownsampler(x)
        x_filter = filter(x) * prefactor

        downsample_kernel(mydev, x_filter, N, down_factor)
    end
end
