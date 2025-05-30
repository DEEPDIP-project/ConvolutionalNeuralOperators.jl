using LuxCore: AbstractLuxLayer
using Random: AbstractRNG
using ComponentArrays: ComponentArray
using Atomix: @atomic
using AbstractFFTs: fft, ifft
using FFTW: fft, ifft

@kernel inbounds = true function convolve_kernel(ffty_r, ffty_im, fft_x, fft_k, ch_x)
    i, j, c, b = @index(Global, NTuple)
    for ci = 1:ch_x
        y = fft_x[i, j, ci, b] * fft_k[c, i, j]
        # In order to use atomic operation I have to split the real and imaginary part
        @atomic ffty_r[i, j, c, b] += real(y)
        @atomic ffty_im[i, j, c, b] += imag(y)
    end
end

function convolve(x, k)
    fft_x = fft(x, (1, 2))
    fft_k = fft(k, (2, 3))

    if CUDA.functional() && k isa CuArray
        # TODO: type is hardcoded
        ffty_r = CUDA.zeros(Float32, size(x, 1), size(x, 2), size(k, 1), size(x, 4))
        ffty_im = CUDA.zeros(Float32, size(x, 1), size(x, 2), size(k, 1), size(x, 4))
        backend = CUDABackend()
        workgroupsize = 256
    else
        ffty_r = zeros(Float32, size(x, 1), size(x, 2), size(k, 1), size(x, 4))
        ffty_im = zeros(Float32, size(x, 1), size(x, 2), size(k, 1), size(x, 4))
        backend = KernelAbstractions.CPU()
        workgroupsize = 64
    end

    # Launch the kernel
    convolve_kernel(backend, workgroupsize)(
        ffty_r,
        ffty_im,
        fft_x,
        fft_k,
        size(x, 3);
        ndrange = size(ffty_r),
    )

    real(ifft(ComplexF32.(ffty_r, ffty_im), (1, 2)))
end


function ChainRulesCore.rrule(::typeof(convolve), x, k)
    # Given Y = X * K (where * denotes convolution),
    # the gradients for backpropagation are:
    #
    # 1. Gradient w.r.t. X:
    #    ∂L/∂X = (∂L/∂Y) * flip(K)
    #    In the Fourier domain: ℱ(∂L/∂X) = ℱ(∂L/∂Y) * conj(ℱ(K))
    #
    # 2. Gradient w.r.t. K:
    #    ∂L/∂K = flip(X * (∂L/∂Y))
    #    In the Fourier domain: ℱ(∂L/∂K) = conj(ℱ(X)) * ℱ(∂L/∂Y)
    #
    # Here, flip(K) represents a 180-degree rotation (flipping in both dimensions),
    # and conj() denotes the complex conjugate in the Fourier domain.

    y = convolve(x, k)
    fft_x = fft(x, (1, 2))
    fft_k = fft(k, (2, 3))

    function convolve_pb(y_bar)
        yb = unthunk(y_bar)
        ffty_bar = fft(yb, (1, 2))

        if CUDA.functional() && k isa CuArray
            x_bar_re = CUDA.zeros(Float32, size(x))
            x_bar_im = CUDA.zeros(Float32, size(x))
            k_bar_re = CUDA.zeros(Float32, size(k))
            k_bar_im = CUDA.zeros(Float32, size(k))
            backend = CUDABackend()
            workgroupsize = 256
        else
            x_bar_re = zeros(Float32, size(x))
            x_bar_im = zeros(Float32, size(x))
            k_bar_re = zeros(Float32, size(k))
            k_bar_im = zeros(Float32, size(k))
            backend = KernelAbstractions.CPU()
            workgroupsize = 64
        end

        # Launch the adjoint kernel for x
        tx_bar = @thunk begin
            convolve_adjoint_x_kernel(backend, workgroupsize)(
                x_bar_re,
                x_bar_im,
                ffty_bar,
                fft_k;
                ndrange = size(x),
            )
            x_bar = ComplexF32.(x_bar_re, x_bar_im)
            x_bar = real(ifft(x_bar, (1, 2)))
        end
        # Launch the adjoint kernel for k
        tk_bar = @thunk begin
            convolve_adjoint_k_kernel(backend, workgroupsize)(
                k_bar_re,
                k_bar_im,
                fft_x,
                ffty_bar,
                size(x, 3);
                ndrange = size(k),
            )
            k_bar = ComplexF32.(k_bar_re, k_bar_im)
            k_bar = real(ifft(k_bar, (2, 3)))
        end

        return NoTangent(), tx_bar, tk_bar
    end
    return y, convolve_pb
end

@kernel inbounds = true function convolve_adjoint_x_kernel(
    x_bar_re,
    x_bar_im,
    ffty_bar,
    fft_k,
)
    i, j, ci, b = @index(Global, NTuple)
    for c = 1:size(fft_k, 1)
        # Use the complex conjugate to backprop the convolution
        y = ffty_bar[i, j, c, b] * conj(fft_k[c, i, j])
        @atomic x_bar_re[i, j, ci, b] += real(y)
        @atomic x_bar_im[i, j, ci, b] += imag(y)
    end
end

@kernel inbounds = true function convolve_adjoint_k_kernel(
    k_bar_re,
    k_bar_im,
    fft_x,
    ffty_bar,
    ch_x,
)
    c, i, j = @index(Global, NTuple)
    for b = 1:size(fft_x, 4)
        for ci = 1:ch_x
            y = conj(fft_x[i, j, ci, b]) * ffty_bar[i, j, c, b]
            @atomic k_bar_re[c, i, j] += real(y)
            @atomic k_bar_im[c, i, j] += imag(y)
        end
    end
end


function apply_masked_convolution(y, k, mask)
    # to get the correct k i have to reshape+mask+trim
    # ! Zygote does not like that you reuse variable names so k2 and k3 needs to be defined
    # ! also Zygote wants the mask to be explicitely defined as a vector so mask_kernel is needed

    # Apply the mask to the kernel
    k2 = mask_kernel(k, mask)

    ## Adjust the kernel size to match the input dimensions
    k3 = trim_kernel(k2, size(y))
    #k3 = k2

    # Apply the convolution
    y = convolve(y, k3)

    return y
end

function trim_kernel(k, sizex)
    xx, xy, _, _ = sizex
    # Trim the kernel to match the input dimensions
    if k isa CuArray
        return CUDA.@allowscalar(k[:, 1:xx, 1:xy])
    else
        return @view k[:, 1:xx, 1:xy]
    end
end

function ChainRulesCore.rrule(::typeof(trim_kernel), k, sizex)
    y = trim_kernel(k, sizex)
    k_bar = similar(k, Float32)

    function trim_kernel_pullback(y_bar)
        yb = unthunk(y_bar)
        sz2, sz3 = size(yb, 2), size(yb, 3)
        k_bar .= 0  # clear first to be safe
        k_bar[:, 1:sz2, 1:sz3] .= yb
        return NoTangent(), k_bar, NoTangent()
    end
    return y, trim_kernel_pullback
end



function mask_kernel(k, mask)
    permutedims(permutedims(k, [2, 3, 1]) .* mask, [3, 1, 2])
end

function get_kernel(ks, chrange)
    if ks isa CuArray
        return CUDA.@allowscalar(ks[chrange, :, :])
    else
        return @view(ks[chrange, :, :])
    end
end

function ChainRulesCore.rrule(::typeof(get_kernel), ks, chrange)
    result = get_kernel(ks, chrange)

    function get_kernel_pullback(result_bar)
        if ks isa CuArray
            k_bar = CUDA.zeros(Float32, size(ks))
            k_bar[chrange, :, :] .= CUDA.@allowscalar(result_bar)
        else
            k_bar = zeros(Float32, size(ks))
            k_bar[chrange, :, :] .= result_bar
        end

        return NoTangent(), k_bar, NoTangent()
    end

    return result, get_kernel_pullback
end
