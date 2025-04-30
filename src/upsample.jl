
function expand_with_zeros(mydev, x, N, up_factor)
    backend = mydev["bck"]
    workgroupsize = mydev["workgroupsize"]
    T = mydev["T"]
    D_up = up_factor * N
    D = ndims(x) - 2
    up_size = (D_up for _ = 1:D)

    if x isa CuArray || (x isa SubArray && parent(x) isa CuArray)
        x_up = CUDA.zeros(T, up_size..., size(x)[end-1], size(x)[end])
    else
        x_up = zeros(T, up_size..., size(x)[end-1], size(x)[end])
    end

    @kernel inbounds = true function expand_kernel!(x, x_up, up_factor)
        i, j, ch, batch = @index(Global, NTuple)
        i_up = (i - 1) * up_factor + 1
        j_up = (j - 1) * up_factor + 1
        x_up[i_up, j_up, ch, batch] = x[i, j, ch, batch]
    end

    expand_kernel!(backend, workgroupsize)(x, x_up, up_factor; ndrange = size(x))
    return x_up
end

function ChainRulesCore.rrule(::typeof(expand_with_zeros), mydev, x, N, up_factor)
    backend = mydev["bck"]
    workgroupsize = mydev["workgroupsize"]
    T = mydev["T"]
    y = expand_with_zeros(mydev, x, N, up_factor)
    function expand_with_zeros_pb(ybar)

        if x isa CuArray || (x isa SubArray && parent(x) isa CuArray)
            xbar = CUDA.zeros(T, size(x))
        else
            xbar = zeros(T, size(x))
        end

        @kernel inbounds = true function expand_kernel_pb!(xbar, ybar, up_factor)
            i, j, ch, batch = @index(Global, NTuple)
            i_up = (i - 1) * up_factor + 1
            j_up = (j - 1) * up_factor + 1
            xbar[i, j, ch, batch] = ybar[i_up, j_up, ch, batch]
        end

        expand_kernel_pb!(backend, workgroupsize)(
            xbar,
            ybar,
            up_factor;
            ndrange = size(xbar),
        )

        return NoTangent(), NoTangent(), xbar, NoTangent(), NoTangent()
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
    force_cpu = false,
)
    D_up = up_factor * N
    grid_up = collect(0.0:(1.0/(D_up-1)):1.0)
    filter =
        create_filter(T, grid_up, cutoff, filter_type = filter_type, force_cpu = force_cpu)

    if CUDA.functional() && !force_cpu
        backend = CUDABackend()
        workgroupsize = 256
    else
        backend = KernelAbstractions.CPU()
        workgroupsize = 64
    end
    mydev = Dict("bck" => backend, "workgroupsize" => workgroupsize, "T" => T)

    function CNOupsampler(x)
        # Enhance to the upsampled size
        x_up = expand_with_zeros(mydev, x, N, up_factor)
        # then apply the lowpass filter
        filter(x_up)
    end
end
