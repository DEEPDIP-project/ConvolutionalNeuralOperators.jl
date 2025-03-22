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
actlayer_identity = create_CNOactivation(
    T,
    D,
    N,
    cutoff,
    activation_function = identity,
    force_cpu = true,
)
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
        layers = (x -> actlayer_tanhshrink(x),)
        closure = Lux.Chain(layers...)
        rng = Random.Xoshiro(123)
        θ, st = Lux.setup(rng, closure)
        θ = ComponentArray(θ)

        # Trigger closure and verify output size
        out = closure(u, θ, st)
        @test size(out[1]) == size(u)

        # Gradient calculation test
        grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
        @test !isnothing(grad)  # Ensure gradient calculation was successful
        @info "grad is $grad"


        # below is the test used for updown: adapt him to above
        ---------------------
        x_filter = rand(Float32, 16, 16, 2, 1)
        result = zeros(Float32, 8, 8, 2, 1)
        down_factor = 2
        mydev = Dict("bck" => CPU(), "workgroupsize" => 64, "T" => Float32)
        downsample_kernel(mydev, x_filter, down_factor, 16)
        @test sum(result) !== 0.0

        y, back = Zygote.pullback(downsample_kernel, mydev, x_filter, 16, down_factor)
        x_filter_bar = zeros(Float32, size(x_filter))
        result_bar = rand(Float32, size(result))
        _, x_filter_bar, _, _ = back(result_bar)
        filtered_size = (((1:down_factor:16) for _ = 1:D)..., :, :)
        redown = x_filter_bar[filtered_size...]
        @test redown == result_bar
    end

end

#@testset "CNO Activation (GPU)" begin
#
#    if !CUDA.functional()
#        @error "CUDA is not functional, skipping test"
#        return
#    end
#
#    # Setup initial image and parameters
#    N0 = 512
#    T = Float32
#    D = 2
#    u0 = zeros(T, N0, N0, D, 1)
#    u0[:, :, 1, 1] .= testimage("cameraman")
#    u0 = CuArray(u0)
#    cutoff = 0.1
#
#    # Test initial image dimensions
#    @test size(u0) == (N0, N0, D, 1)
#
#    # Downsize the input
#    down_factor = 6
#    ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
#    u = ds(u0)
#    N = size(u, 1)
#
#    # Test downsampled size
#    @test size(u) == (N, N, D, 1)
#    @test N - 1 == div(N0, down_factor)
#
#    # Test create_CNOactivation with identity activation
#    al_identity = create_CNOactivation(T, D, N, cutoff, activation_function = identity)
#    u_identity = al_identity(u)
#    @test size(u_identity) == size(u)
#
#    # Test create_CNOactivation with tanhshrink activation
#    al_tanhshrink = create_CNOactivation(T, D, N, cutoff, activation_function = tanhshrink)
#    u_tanhshrink = al_tanhshrink(u)
#    @test size(u_tanhshrink) == size(u)
#
#    # Differentiability Test
#    layers = (x -> al_tanhshrink(x),)
#    closure = Lux.Chain(layers...)
#    rng = Random.Xoshiro(123)
#    θ, st = Lux.setup(rng, closure)
#    dev = Lux.gpu_device()
#    θ = ComponentArray(θ)
#
#    # Trigger closure and verify output size
#    out = closure(u, θ, st)
#    @test size(out[1]) == size(u)
#
#    # Gradient calculation test
#    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    @test !isnothing(grad)  # Ensure gradient calculation was successful
#
#end
#