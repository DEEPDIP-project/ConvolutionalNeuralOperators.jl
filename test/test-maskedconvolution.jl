using DifferentialEquations: ODEProblem, solve
using Optimization: Optimization, solve
using OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
using Random: Random
using TestImages: testimage
using ComponentArrays: ComponentArray
using Lux: Lux
using CUDA
using LuxCUDA
using ConvolutionalNeuralOperators: convolve, apply_masked_convolution
using Zygote: Zygote
using Test  # Importing the Test module for @test statements
using AbstractFFTs: fft, ifft
using FFTW: fft, ifft
using ChainRulesCore

rng = Random.Xoshiro(123)


@testset "Masked-Convolution (CPU)" begin
    @testset "Forward" begin
        x = rand(Float32, 16, 16, 2, 1)
        k = rand(Float32, 5, 16, 16)
        mask = ones(Float32, 16, 16)

        y = apply_masked_convolution(x, k, mask)
        y_ref = convolve(x, k)
        @test size(y) == size(y_ref)
        @test y ≈ y_ref

        mask = rand(Float32, 16, 16)
        y = apply_masked_convolution(x, k, mask)
        @test size(y) == size(y_ref)
        @test y !== y_ref
        @test maximum(abs.(y - y_ref)) > 1.0

    end

    @testset "AD" begin
        x = rand(Float32, 16, 16, 2, 4)
        k = rand(Float32, 5, 16, 16)
        mask = ones(Float32, 16, 16)

        y_ref, back_ref = Zygote.pullback(convolve, x, k)
        y, back = Zygote.pullback(apply_masked_convolution, x, k, mask)
        @test y ≈ y_ref
        y_bar = rand(Float32, size(y))
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        x_bar, k_bar, _ = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar) !== 0.0
        @test x_bar ≈ x_bar_ref
        @test k_bar ≈ k_bar_ref

        mask = rand(Float32, 16, 16)
        y, back = Zygote.pullback(apply_masked_convolution, x, k, mask)
        y_ref, back_ref = Zygote.pullback(convolve, x, k)
        y_bar = rand(Float32, size(y))
        x_bar, k_bar, _ = back(y_bar)
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        @test maximum(abs.(y - y_ref)) > 1.0
        @test maximum(abs.(x_bar - x_bar_ref)) > 1.0
        @test maximum(abs.(k_bar - k_bar_ref)) > 1.0

    end

end


@testset "Masked-Convolution (GPU)" begin
    @testset "Forward" begin
        x = CUDA.rand(Float32, 16, 16, 2, 1)
        k = CUDA.rand(Float32, 5, 16, 16)
        mask = CUDA.ones(Float32, 16, 16)

        y = apply_masked_convolution(x, k, mask)
        y_ref = convolve(x, k)
        @test size(y) == size(y_ref)
        @test y ≈ y_ref

        mask = CUDA.rand(Float32, 16, 16)
        y = apply_masked_convolution(x, k, mask)
        @test size(y) == size(y_ref)
        @test y !== y_ref
        @test maximum(abs.(y - y_ref)) > 1.0

    end

    @testset "AD" begin
        x = CUDA.rand(Float32, 16, 16, 2, 4)
        k = CUDA.rand(Float32, 5, 16, 16)
        mask = CUDA.ones(Float32, 16, 16)

        y_ref, back_ref = Zygote.pullback(convolve, x, k)
        y, back = Zygote.pullback(apply_masked_convolution, x, k, mask)
        @test y ≈ y_ref
        y_bar = CUDA.rand(Float32, size(y))
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        x_bar, k_bar, _ = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar) !== 0.0
        @test x_bar ≈ x_bar_ref
        @test k_bar ≈ k_bar_ref

        mask = CUDA.rand(Float32, 16, 16)
        y, back = Zygote.pullback(apply_masked_convolution, x, k, mask)
        y_ref, back_ref = Zygote.pullback(convolve, x, k)
        y_bar = CUDA.rand(Float32, size(y))
        x_bar, k_bar, _ = back(y_bar)
        x_bar_ref, k_bar_ref = back_ref(y_bar)
        @test maximum(abs.(y - y_ref)) > 1.0
        @test maximum(abs.(x_bar - x_bar_ref)) > 1.0
        @test maximum(abs.(k_bar - k_bar_ref)) > 1.0

    end

end
