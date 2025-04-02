using Test
using Adapt
using TestImages: testimage
using Random: Random
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO
using NNlib: tanhshrink
using Lux
using LuxCUDA
using Zygote: Zygote
using ComponentArrays: ComponentArray
using CairoMakie: Figure, Axis, heatmap, save, heatmap!, GridLayout
using Images: load
using CUDA


# Setup initial image and parameters
N0 = 512
T = Float32
D = 2
u0 = zeros(T, N0, N0, D, 1)
u0[:, :, 1, 1] .= testimage("cameraman")
cutoff = 0.1
# Downsize the input
down_factor = 6
ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff, force_cpu = true)
u = ds(u0)
N = size(u, 1)
# Define some activation layers
# (1) Identity activation
actlayer_identity =
    create_CNOactivation(T, D, N, cutoff, activation_function = identity, force_cpu = true)
u_identity = actlayer_identity(u)
# (2) Tanhshrink activation
actlayer_tanhshrink = create_CNOactivation(
    T,
    D,
    N,
    cutoff,
    activation_function = tanhshrink,
    force_cpu = true,
)
u_tanhshrink = actlayer_tanhshrink(u)

@testset "CNO Activation (CPU)" begin

    @testset "Initial Image Dimensions" begin
        @test size(u0) == (N0, N0, D, 1)
        @test size(u) == (N, N, D, 1)
        @test N == div(N0, down_factor)
    end


    @testset "Identity activation" begin
        @test size(u_identity) == size(u)
    end

    # Test create_CNOactivation with tanhshrink activation
    @testset "Tanhshrink activation" begin
        @test size(u_tanhshrink) == size(u)
    end


    # Visualization tests
    @testset "Visualization Tests" begin
        if Sys.KERNEL == "Darwin"
            @info "Skipping Visualization Tests on macOS"
            return
        end
        fig = Figure(resolution = (800, 400))
        ax1 = Axis(fig[1, 1], title = "Identity Activation on u")
        heatmap!(ax1, u_identity[:, :, 1, 1])
        ax2 = Axis(fig[1, 2], title = "Tanhshrink Activation on u")
        heatmap!(ax2, u_tanhshrink[:, :, 1, 1])

        save("test_figs/activation.png", fig)
        img1 = load("test_figs/activation.png")
        img2 = load("test_figs/activation_baseline.png")
        @test img1 == img2
    end

    @testset "Activation AD" begin
        result = actlayer_tanhshrink(u)
        @test sum(result) !== 0.0

        y, back = Zygote.pullback(actlayer_tanhshrink, u)
        @test y == result
        y_bar = rand(Float32, size(u))
        x_bar = zeros(Float32, size(u))
        x_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test x_bar != y_bar
    end

end


@testset "CNO Activation (GPU)" begin
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    CUDA.allowscalar(false)

    # Prepare for GPU tests
    u = CuArray(u)
    actlayer_identity =
        create_CNOactivation(T, D, N, cutoff, activation_function = identity)
    u_identity = actlayer_identity(u)
    actlayer_tanhshrink =
        create_CNOactivation(T, D, N, cutoff, activation_function = tanhshrink)
    u_tanhshrink = actlayer_tanhshrink(u)


    @testset "Initial Image Dimensions" begin
        @test size(u0) == (N0, N0, D, 1)
        @test size(u) == (N, N, D, 1)
        @test N == div(N0, down_factor)
    end


    @testset "Identity activation" begin
        @test isa(u_identity, CuArray)
        @test size(u_identity) == size(u)
    end

    # Test create_CNOactivation with tanhshrink activation
    @testset "Tanhshrink activation" begin
        @test isa(u_tanhshrink, CuArray)
        @test size(u_tanhshrink) == size(u)
    end


    @testset "Activation AD identity" begin
        result = actlayer_identity(u)
        @test sum(result) !== 0.0

        y, back = Zygote._pullback(actlayer_identity, u)
        @test y == result
        y_bar = CUDA.rand(Float32, size(u))
        x_bar = CUDA.zeros(Float32, size(u))
        _, x_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test x_bar != y_bar
        @test isa(x_bar, CuArray)
        @test isa(y, CuArray)
    end

    @testset "Activation AD tanhshrink" begin
        result = actlayer_tanhshrink(u)
        @test sum(result) !== 0.0

        y, back = Zygote._pullback(actlayer_tanhshrink, u)
        @test y == result
        y_bar = CUDA.rand(Float32, size(u))
        x_bar = CUDA.zeros(Float32, size(u))
        @test isa(x_bar, CuArray)
        @test isa(y_bar, CuArray)
        _, x_bar = back(y_bar)
        @test sum(x_bar) !== 0.0
        @test x_bar != y_bar
        @test isa(x_bar, CuArray)
        @test isa(y, CuArray)
    end

end
