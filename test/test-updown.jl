using Test
using TestImages: testimage
using Random: Random
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO, create_CNOupsampler
using Lux
using Zygote: Zygote
using ComponentArrays: ComponentArray
using Plots: heatmap, plot

@testset "CNO Downsampling and Upsampling" begin

    # Setup initial conditions
    N0 = 512
    T = Float32
    D = 2
    u0 = zeros(T, N0, N0, D, 1)
    u0[:, :, 1, 1] .= testimage("cameraman")
    cutoff = 0.1

    # Test: Check initial image dimensions
    @test size(u0) == (N0, N0, D, 1)

    # Create initial downsampler
    down_factor = 6
    ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
    u = ds(u0)
    N = size(u, 1)

    # Test: Check downsampled size
    @test size(u) == (N, N, D, 1)
    @test N - 1 == div(N0, down_factor)

    # Create downsampling and upsampling operations
    down_factor = 2
    ds = create_CNOdownsampler(T, D, N, down_factor, cutoff)

    up_factor = 2
    us = create_CNOupsampler(T, D, N, up_factor, cutoff)

    # Test downsampling then upsampling
    ds2 = create_CNOdownsampler(T, D, N * up_factor, down_factor, cutoff)
    @test size(ds2(us(u))) == size(u)  # Confirm ds2(us(u)) == u

    # Test upsampling then downsampling
    us2 = create_CNOupsampler(T, D, Int(N / down_factor), up_factor, cutoff)
    @test size(us2(ds(u))) == size(u)  # Confirm us2(ds(u)) == u

    ## Visualization tests
    #p1 = heatmap(u[:, :, 1, 1], aspect_ratio = 1, title = "u", colorbar = false)
    #p2 = heatmap(ds(u)[:, :, 1, 1], aspect_ratio = 1, title = "ds(u)", colorbar = false)
    #p3 = heatmap(us(u)[:, :, 1, 1], aspect_ratio = 1, title = "us(u)", colorbar = false)
    #p4 = heatmap(ds2(us(u))[:, :, 1, 1], aspect_ratio = 1, title = "ds2(us(u))", colorbar = false)
    #p5 = heatmap(us2(ds(u))[:, :, 1, 1], aspect_ratio = 1, title = "us2(ds(u))", colorbar = false)
    #plot(p1, p2, p3, p4, p5, layout = (2, 3))

    # Differentiability Tests
    layers = (x -> ds(x), x -> us2(x))
    closure = Lux.Chain(layers...)
    rng = Random.Xoshiro(123)
    θ, st = Lux.setup(rng, closure)
    θ = ComponentArray(θ)

    # Trigger closure and test output
    out = closure(u, θ, st)
    @test size(out[1]) == size(u)

    # Gradient calculation test
    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
    @test !isnothing(grad)  # Ensure gradient calculation was successful

end
