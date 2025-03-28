using DifferentialEquations: ODEProblem, solve
using Optimization: Optimization, solve
using OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
using Random: Random
using TestImages: testimage
using ComponentArrays: ComponentArray
using Lux: Lux
using CUDA
using LuxCUDA
using ConvolutionalNeuralOperators:
    convolve, apply_masked_convolution, trim_kernel, get_kernel
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

    @testset "Get kernel" begin
        ks = rand(Float32, 30, 16, 16)
        k = get_kernel(ks, 5:10)
        @test size(k) == (6, 16, 16)

        y_bar = ones(Float32, size(k))
        k, back = Zygote.pullback(get_kernel, ks, 5:10)
        ks_bar, _ = back(y_bar)
        @test all(ks_bar[1:4, :, :] .== 0.0)
        @test all(ks_bar[5:10, :, :] .== 1.0)
        @test all(ks_bar[11:end, :, :] .== 0.0)
    end


    @testset "Kernel trim" begin
        x = rand(Float32, 16, 16, 2, 4)
        k = rand(Float32, 5, 64, 64)
        y = trim_kernel(k, size(x))
        @test size(y) == (5, 16, 16)

        y_bar = ones(Float32, size(y))
        y, back = Zygote.pullback(trim_kernel, k, size(x))
        x_bar, _ = back(y_bar)
        @test all(x_bar[1, 1:16, 1:16] .== 1.0)
        @test all(x_bar[1, 17:end, 17:end] .== 0.0)
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
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    CUDA.allowscalar(false)
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

    @testset "Get kernel" begin
        ks = CUDA.rand(Float32, 30, 16, 16)
        k = get_kernel(ks, 5:11)
        @test size(k) == (7, 16, 16)

        y_bar = CUDA.ones(Float32, size(k))
        k, back = Zygote.pullback(get_kernel, ks, 5:11)
        ks_bar, _ = back(y_bar)
        @test all(ks_bar[1:4, :, :] .== 0.0)
        @test all(ks_bar[5:11, :, :] .== 1.0)
        @test all(ks_bar[12:end, :, :] .== 0.0)
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
        @test isa(x_bar, CUDA.CuArray)
        @test isa(k_bar, CUDA.CuArray)
        @test isa(y, CUDA.CuArray)

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
