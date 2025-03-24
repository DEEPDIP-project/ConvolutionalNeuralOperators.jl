using Lux: Lux, relu, leakyrelu
using LuxCUDA
using LuxCore: AbstractLuxLayer
using Random: AbstractRNG
using ComponentArrays: ComponentArray
using KernelAbstractions
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
        backend = CPU()
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
        ffty_bar = fft(y_bar, (1, 2))

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
            backend = CPU()
            workgroupsize = 64
        end

        # Launch the adjoint kernel for x
        convolve_adjoint_x_kernel(backend, workgroupsize)(
            x_bar_re,
            x_bar_im,
            ffty_bar,
            fft_k;
            ndrange = size(x),
        )
        # Launch the adjoint kernel for k
        convolve_adjoint_k_kernel(backend, workgroupsize)(
            k_bar_re,
            k_bar_im,
            fft_x,
            ffty_bar,
            size(x, 3);
            ndrange = size(k),
        )

        x_bar = ComplexF32.(x_bar_re, x_bar_im)
        k_bar = ComplexF32.(k_bar_re, k_bar_im)

        x_bar = real(ifft(x_bar, (1, 2)))
        k_bar = real(ifft(k_bar, (2, 3)))

        return NoTangent(), x_bar, k_bar
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
