using Test
using TestImages: testimage
using Random: Random
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO, create_CNOupsampler, create_filter, downsample_kernel!
using Lux
using CUDA
using LuxCUDA
using Zygote: Zygote
using ComponentArrays: ComponentArray
using CairoMakie: Figure, Axis, heatmap, save, heatmap!, GridLayout
using Images: load
using KernelAbstractions
using ChainRulesTestUtils
using ChainRulesCore


# Setup initial conditions
N0 = 512
T = Float32
D = 2
u0 = zeros(T, N0, N0, D, 1)
u0[:, :, 1, 1] .= testimage("cameraman")
cutoff = 0.1
# first scale down to work with smaller images
down_factor0 = 8
ds0 = create_CNOdownsampler(T, D, N0, down_factor0, cutoff, force_cpu = true)
u = ds0(u0)
N = size(u, 1)
# and ds and us layers
down_factor = 2
ds = create_CNOdownsampler(T, D, N, down_factor, cutoff, force_cpu = true)
up_factor = 2
us = create_CNOupsampler(T, D, N, up_factor, cutoff, force_cpu = true)
ds2 = create_CNOdownsampler(T, D, N * up_factor, down_factor, cutoff, force_cpu = true)
us2 = create_CNOupsampler(T, D, Int(N / down_factor), up_factor, cutoff, force_cpu = true)

@testset "CNO Downsampling and Upsampling (CPU)" begin

    @testset "Initial Image Dimensions" begin
        @test size(u0) == (N0, N0, D, 1)
        @test size(u) == (N, N, D, 1)
        @test N == div(N0, down_factor0)
    end

    @testset "Downsampling and Upsampling Operations" begin
        @test size(ds2(us(u))) == size(u)
        @test size(us2(ds(u))) == size(u)
    end

    @testset "Visualization Tests" begin
        fig = Figure(resolution = (800, 600))
        ax1 = Axis(fig[1, 1], title = "u")
        heatmap!(ax1, u[:, :, 1, 1])
        ax2 = Axis(fig[1, 2], title = "ds(u)")
        heatmap!(ax2, ds(u)[:, :, 1, 1])
        ax3 = Axis(fig[1, 3], title = "us(u)")
        heatmap!(ax3, us(u)[:, :, 1, 1])
        ax4 = Axis(fig[2, 1], title = "ds2(us(u))")
        heatmap!(ax4, ds2(us(u))[:, :, 1, 1])
        ax5 = Axis(fig[2, 2], title = "us2(ds(u))")
        heatmap!(ax5, us2(ds(u))[:, :, 1, 1])
        fig[2, 3] = GridLayout()
        save("test_figs/downsampling_upsampling.png", fig)
        img1 = load("test_figs/downsampling_upsampling.png")
        img2 = load("test_figs/downsampling_upsampling_baseline.png")
        @test img1 == img2
    end

    @testset "Direct Downsampler comparison" begin
        function direct_downsampler(
            T::Type,
            D::Int,
            N::Int,
            down_factor::Int,
            cutoff,
            filter_type = "sinc";
        )
            grid = collect(0.0:(1.0/(N-1)):1.0)
            filtered_size = (((1:down_factor:N) for _ = 1:D)..., :, :)
            filter = create_filter(T, grid, cutoff, filter_type = filter_type, force_cpu = true)
            prefactor = T(1 / down_factor^D)

            function CNOdownsampler(x)
                x_filter = filter(x) * prefactor
                x_filter[filtered_size...]
            end
        end
        direct_ds = direct_downsampler(T, D, N, down_factor, cutoff)
        @test direct_ds(u) ≈ ds(u)
    end

    @testset "Downsample Kernel Test" begin
        x_filter = rand(Float32, 16, 16, 2, 1)
        result = zeros(Float32, 8, 8, 2, 1)
        down_factor = 2
        mydev = Dict("bck" => CPU(), "workgroupsize" => 64)
        downsample_kernel!(mydev, x_filter, result, down_factor, (8, 8, 2, 1))
        @test sum(result) !== 0.0

#        layers = (x -> begin result = zeros(Float32, 8, 8, 2, 1); downsample_kernel!(CPU(), 64, x, result, down_factor, (8, 8, 2, 1)); result end)
#        closure = Lux.Chain(layers)
#        rng = Random.Xoshiro(123)
#        θ, st = Lux.setup(rng, closure)
#        θ = ComponentArray(θ)
#        out = closure(x_filter, θ, st)
#        #grad = Zygote.gradient(θ -> closure(x_filter, θ, st)[1][1], θ)
#        grad = Zygote.gradient(x_filter -> closure(x_filter, θ, st)[1][1].*2, x_filter)
#        @test !isnothing(grad)
#        @info "Gradient computed: $grad"

        y, back = Zygote.pullback(downsample_kernel!, mydev, x_filter, result, down_factor, (8, 8, 2, 1))
        x_filter_bar = ones(Float32, size(x_filter))
        result_bar = zeros(Float32, size(result))
        back(result_bar)
        @test sum(result_bar) !== 0.0
        
        @info length(mydev)
        @info length(x_filter)
        @info length(result)
        @info length(down_factor)
        @info length((8, 8, 2, 1))
        test_rrule(downsample_kernel!, mydev⊢ NoTangent(), x_filter, result, down_factor⊢ NoTangent(), (8, 8, 2, 1)⊢ NoTangent())
    end

end
@testset "CNO Downsampling and Upsampling" begin


#    # ************************************************************************
#    # Differentiability Tests (down + up)
#    layers = (x -> ds(x), x -> us2(x))
#    closure = Lux.Chain(layers...)
#    rng = Random.Xoshiro(123)
#    θ, st = Lux.setup(rng, closure)
#    θ = ComponentArray(θ)
#    out = closure(u, θ, st)
#    @test size(out[1]) == size(u)
#    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    @test !isnothing(grad)  # Ensure gradient calculation was successful
#
#
    # ************************************************************************
    # Differentiability Tests (down only)
#    layers = (x -> ds(x))
#    closure = Lux.Chain(layers)
#    rng = Random.Xoshiro(123)
#    θ, st = Lux.setup(rng, closure)
#    θ = ComponentArray(θ)
#    out = closure(u, θ, st)
#    @test size(out[1]) == (32,32,2,1)
#    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    @test !isnothing(grad)  # Ensure gradient calculation was successful



#    # ************************************************************************
#    # Differentiability Tests (up only)
#    layers = (x -> us(x))
#    closure = Lux.Chain(layers)
#    rng = Random.Xoshiro(123)
#    θ, st = Lux.setup(rng, closure)
#    θ = ComponentArray(θ)
#    out = closure(u, θ, st)
#    @test size(out[1]) == (172,172,2,1)
#    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    @test !isnothing(grad)  # Ensure gradient calculation was successful
#
end

#@testset "CNO Downsampling and Upsampling (GPU)" begin
#
#    # Setup initial conditions
#    N0 = 512
#    T = Float32
#    D = 2
#    u0 = zeros(T, N0, N0, D, 1)
#    u0[:, :, 1, 1] .= testimage("cameraman")
#    u0 = CuArray(u0)
#    cutoff = 0.1
#
#
#    # Create initial downsampler
#    down_factor = 6
#    ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
#    u = ds(u0)
#    N = size(u, 1)
#
#    # Test: Check downsampled size
#    @test size(u) == (N, N, D, 1)
#    @test N - 1 == div(N0, down_factor)
#
#    # Create downsampling and upsampling operations
#    down_factor = 2
#    ds = create_CNOdownsampler(T, D, N, down_factor, cutoff)
#
#    up_factor = 2
#    us = create_CNOupsampler(T, D, N, up_factor, cutoff)
#
#    # Test downsampling then upsampling
#    ds2 = create_CNOdownsampler(T, D, N * up_factor, down_factor, cutoff)
#    @test size(ds2(us(u))) == size(u)  # Confirm ds2(us(u)) == u
#
#    # Test upsampling then downsampling
#    us2 = create_CNOupsampler(T, D, Int(N / down_factor), up_factor, cutoff)
#    @test size(us2(ds(u))) == size(u)  # Confirm us2(ds(u)) == u
#
#    ## Differentiability Tests
#    #layers = (x -> ds(x), x -> us2(x))
#    #closure = Lux.Chain(layers...)
#    #rng = Random.Xoshiro(123)
#    #θ, st = Lux.setup(rng, closure)
#    #θ = ComponentArray(θ)
#
#    ## Trigger closure and test output
#    #out = closure(u, θ, st)
#    #@test size(out[1]) == size(u)
#
#    ## Gradient calculation test
#    #grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    #@test !isnothing(grad)  # Ensure gradient calculation was successful
#
#
#    # Differentiability Tests (down only)
#    layers = (x -> ds(x))
#    closure = Lux.Chain(layers)
#    rng = Random.Xoshiro(123)
#    θ, st = Lux.setup(rng, closure)
#    θ = ComponentArray(θ)
#    out = closure(u, θ, st)
#    @test size(out[1]) == (43,43,2,1)
#    grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    @test !isnothing(grad)  # Ensure gradient calculation was successful
#
#
#    ## Differentiability Tests (up only)
#    #layers = (x -> us(x))
#    #closure = Lux.Chain(layers)
#    #rng = Random.Xoshiro(123)
#    #θ, st = Lux.setup(rng, closure)
#    #θ = ComponentArray(θ)
#    #out = closure(u, θ, st)
#    #@test size(out[1]) == (172,172,2,1)
#    #grad = Zygote.gradient(θ -> closure(u, θ, st)[1][1], θ)
#    #@test !isnothing(grad)  # Ensure gradient calculation was successful
#
#end