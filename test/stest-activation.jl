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
using Plots: heatmap, plot
using CUDA

@testset "CNO Activation" begin

    # Setup initial image and parameters
    N0 = 512
    T = Float32
    D = 2
    u0 = zeros(T, N0, N0, D, 1)
    u0[:, :, 1, 1] .= testimage("cameraman")
    cutoff = 0.1

    # Test initial image dimensions
    @test size(u0) == (N0, N0, D, 1)

    # Downsize the input
    down_factor = 6
    ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff, force_cpu = true)
    u = ds(u0)
    N = size(u, 1)

    # Test downsampled size
    @test size(u) == (N, N, D, 1)
    @test N - 1 == div(N0, down_factor)

    # Test create_CNOactivation with identity activation
    al_identity = create_CNOactivation(
        T,
        D,
        N,
        cutoff,
        activation_function = identity,
        force_cpu = true,
    )
    u_identity = al_identity(u)
    @test size(u_identity) == size(u)

    # Test create_CNOactivation with tanhshrink activation
    al_tanhshrink = create_CNOactivation(
        T,
        D,
        N,
        cutoff,
        activation_function = tanhshrink,
        force_cpu = true,
    )
    u_tanhshrink = al_tanhshrink(u)
    @test size(u_tanhshrink) == size(u)

    # Visualization tests
    #p1 = heatmap(u_identity[:, :, 1, 1], aspect_ratio = 1, title = "Identity Activation on u", colorbar = false)
    #p2 = heatmap(u_tanhshrink[:, :, 1, 1], aspect_ratio = 1, title = "Tanhshrink Activation on u", colorbar = false)
    #plot(p1, p2, layout = (1, 2))

    # Differentiability Test
    layers = (x -> al_tanhshrink(x),)
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

end

@testset "CNO Activation (GPU)" begin

    if !CUDA.functional()
        @error "CUDA is not functional, skipping test"
        return
    end

    # Setup initial image and parameters
    N0 = 512
    T = Float32
    D = 2
    u0 = zeros(T, N0, N0, D, 1)
    u0[:, :, 1, 1] .= testimage("cameraman")
    u0 = CuArray(u0)
    cutoff = 0.1

    # Test initial image dimensions
    @test size(u0) == (N0, N0, D, 1)

    # Downsize the input
    down_factor = 6
    ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
    u = ds(u0)
    N = size(u, 1)

    # Test downsampled size
    @test size(u) == (N, N, D, 1)
    @test N - 1 == div(N0, down_factor)

    # Test create_CNOactivation with identity activation
    al_identity = create_CNOactivation(T, D, N, cutoff, activation_function = identity)
    u_identity = al_identity(u)
    @test size(u_identity) == size(u)

    # Test create_CNOactivation with tanhshrink activation
    al_tanhshrink = create_CNOactivation(T, D, N, cutoff, activation_function = tanhshrink)
    u_tanhshrink = al_tanhshrink(u)
    @test size(u_tanhshrink) == size(u)

    # Differentiability Test
    layers = (x -> al_tanhshrink(x),)
    closure = Lux.Chain(layers...)
    rng = Random.Xoshiro(123)
    θ, st = Lux.setup(rng, closure)
    dev = Lux.gpu_device()
    θ = ComponentArray(θ)

    # Trigger closure and verify output size
    out = closure(u, θ, st)
    @test size(out[1]) == size(u)

    # Gradient calculation test
    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
    @test !isnothing(grad)  # Ensure gradient calculation was successful

end
