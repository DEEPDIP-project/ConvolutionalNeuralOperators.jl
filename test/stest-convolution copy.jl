using DifferentialEquations: ODEProblem, solve
using Optimization: Optimization, solve
using OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
using Random: Random
using TestImages: testimage
using ComponentArrays: ComponentArray
using Lux: Lux
using CUDA
using LuxCUDA
using ConvolutionalNeuralOperators: convolve
using Zygote: Zygote
using Test  # Importing the Test module for @test statements
using AbstractFFTs: fft, ifft
using FFTW: fft, ifft
using ChainRulesCore

rng = Random.Xoshiro(123)
function reference_convolve(x, k)
    if k isa SubArray && parent(k) isa CuArray
        k = CuArray(collect(k))
    end
    if x isa SubArray && parent(x) isa CuArray
        x = CuArray(collect(x))
    end
    fft_x = fft(x, (1, 2))
    fft_k = fft(k, (2, 3))
    #    ffty = zeros(ComplexF32, size(x, 1), size(x, 2), size(k, 1), size(x, 4))
    #    for c = 1:size(k, 1)
    #        for ci = 1:size(x, 3)
    #            ffty[:,:,c,:] .+= fft_x[:, :, ci, :] .* fft_k[c, :, :]
    #        end
    #    end

    # Can not use for loops if you want it to be differentiable
    ffty = [
        reduce(+, [fft_x[:, :, ci, :] .* fft_k[c, :, :] for ci = 1:size(x, 3)]) for
        c = 1:size(k, 1)
    ]
    ffty = cat(ffty..., dims = 3)
    ffty = permutedims(
        reshape(ffty, size(fft_x)[1], size(fft_x)[2], size(x)[4], size(k)[1]),
        [1, 2, 4, 3],
    )


    real(ifft(ffty, (1, 2)))
end

@testset "Convolution (CPU)" begin
    @testset "Forward" begin
        x = ones(Float32, 16, 16, 2, 1)
        k = zeros(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test size(y) == size(y_ref)
        @test y ≈ y_ref

        x = ones(Float32, 16, 16, 2, 1)
        k = ones(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test y ≈ y_ref

        x = rand(Float32, 16, 16, 2, 4)
        k = ones(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test size(y) == size(y_ref)
        @test y ≈ y_ref

        x = rand(Float32, 16, 16, 2, 4)
        k = rand(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test y ≈ y_ref
    end

    @testset "AD" begin
        # Can not use test_rrule because fft complains about ChainRules types

        x = rand(Float32, 16, 16, 2, 4)
        k = rand(Float32, 5, 16, 16)

        y_ref, back_ref = Zygote.pullback(reference_convolve, x, k)
        y, back = Zygote.pullback(convolve, x, k)
        @test y_ref == convolve(x, k)
        @test y ≈ y_ref
        y_bar = rand(Float32, size(y))
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        x_bar, k_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar) !== 0.0
        @test x_bar ≈ x_bar_ref
        @test k_bar ≈ k_bar_ref


        y_bar = rand(Float32, size(y))
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        x_bar, k_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar) !== 0.0
        @test x_bar ≈ x_bar_ref
        @test k_bar ≈ k_bar_ref

    end

end

@testset "Convolution (GPU)" begin
    @testset "Forward" begin
        x = CUDA.ones(Float32, 16, 16, 2, 1)
        k = CUDA.zeros(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test size(y) == size(y_ref)
        @test y ≈ y_ref

        x = CUDA.ones(Float32, 16, 16, 2, 1)
        k = CUDA.ones(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test y ≈ y_ref

        x = CUDA.rand(Float32, 16, 16, 2, 4)
        k = CUDA.ones(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test size(y) == size(y_ref)
        @test y ≈ y_ref

        x = CUDA.rand(Float32, 16, 16, 2, 4)
        k = CUDA.rand(Float32, 5, 16, 16)
        y_ref = reference_convolve(x, k)
        y = convolve(x, k)
        @test y ≈ y_ref
    end

    @testset "AD" begin
        x = CUDA.rand(Float32, 16, 16, 2, 4)
        k = CUDA.rand(Float32, 5, 16, 16)

        y_ref, back_ref = Zygote.pullback(reference_convolve, x, k)
        y, back = Zygote.pullback(convolve, x, k)
        @test y_ref ≈ convolve(x, k)
        @test y ≈ y_ref
        y_bar = CUDA.rand(Float32, size(y))
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        x_bar, k_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar) !== 0.0
        @test x_bar ≈ x_bar_ref
        @test k_bar ≈ k_bar_ref


        y_bar = CUDA.rand(Float32, size(y))
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        x_bar, k_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar) !== 0.0
        @test x_bar ≈ x_bar_ref
        @test k_bar ≈ k_bar_ref

    end

end
