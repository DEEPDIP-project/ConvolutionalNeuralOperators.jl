using Test
using TestImages: testimage
using Random: Random
using ConvolutionalNeuralOperators:
    create_CNOdownsampler, create_CNO, create_CNOupsampler, create_filter, downsample_kernel
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

    @testset "Upsampler implementation" begin
        function direct_upsampler(
            T::Type,
            D::Int,
            N::Int,
            up_factor::Int,
            cutoff,
            filter_type = "sinc",
        )
            D_up = up_factor * N
            up_size = (D_up for _ = 1:D)
            grid_up = collect(0.0:(1.0/(D_up-1)):1.0)
            filter = create_filter(
                T,
                grid_up,
                cutoff,
                filter_type = filter_type,
                force_cpu = true,
            )

            function expand_with_zeros(x, T, up_size, up_factor)
                x_up = zeros(T, up_size..., size(x)[end-1], size(x)[end])
                x_up[1:up_factor:end, 1:up_factor:end, :, :] .= x
                return x_up
            end

            function CNOupsampler(x)
                # Enhance to the upsampled size
                x_up = expand_with_zeros(x, T, up_size, up_factor)
                # then apply the lowpass filter
                filter(x_up)
            end
        end

        direct_ds = direct_upsampler(T, D, N, up_factor, cutoff)
        uu = direct_ds(u)
        usu = us(u)
        @test uu ≈ usu
    end

    @testset "Downsampler implementation" begin
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
            filter =
                create_filter(T, grid, cutoff, filter_type = filter_type, force_cpu = true)
            prefactor = T(1 / down_factor^D)

            function CNOdownsampler(x)
                x_filter = filter(x) * prefactor
                x_filter[filtered_size...]
            end
        end
        direct_ds = direct_downsampler(T, D, N, down_factor, cutoff)
        @test direct_ds(u) ≈ ds(u)
    end

    @testset "Downsample Kernel AD" begin
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
    @testset "Downsample AD" begin
        u_test = ones(Float32, size(u))
        du = ds(u_test)
        y, back = Zygote._pullback(ds, u_test)
        _, du_bar = back(du)
        @test size(du_bar) == size(u_test)
        @test sum(du_bar) !== 0.0
    end

    @testset "Upsample AD" begin
        u_test = ones(Float32, size(u))
        du = us(u_test)
        y, back = Zygote._pullback(us, u_test)
        _, du_bar = back(du)
        @test size(du_bar) == size(u_test)
        @test sum(du_bar) !== 0.0
    end

    @testset "Down->Up AD" begin
        layers = (x -> ds(x), x -> us2(x))
        closure = Lux.Chain(layers...)
        rng = Random.Xoshiro(123)
        θ, st = Lux.setup(rng, closure)
        θ = ComponentArray(θ)
        function downup(u)
            closure(u, θ, st)[1]
        end
        u_test = ones(Float32, size(u))
        du = downup(u_test)
        @test size(du) == size(u_test)
        y, back = Zygote._pullback(downup, u_test)
        _, du_bar = back(du)
        @test size(du_bar) == size(u_test)
        @test sum(du_bar) !== 0.0

    end
end

# Make into GPU
u0 = CuArray(u0)
ds0 = create_CNOdownsampler(T, D, N0, down_factor0, cutoff)
u = ds0(u0)
ds = create_CNOdownsampler(T, D, N, down_factor, cutoff)
us = create_CNOupsampler(T, D, N, up_factor, cutoff)
ds2 = create_CNOdownsampler(T, D, N * up_factor, down_factor, cutoff)
us2 = create_CNOupsampler(T, D, Int(N / down_factor), up_factor, cutoff)

@testset "CNO Downsampling and Upsampling (GPU)" begin

    @testset "Initial Image Dimensions" begin
        @test size(u0) == (N0, N0, D, 1)
        @test size(u) == (N, N, D, 1)
        @test N == div(N0, down_factor0)
    end

    @testset "Downsampling and Upsampling Operations" begin
        @test size(ds2(us(u))) == size(u)
        @test size(us2(ds(u))) == size(u)
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
            filter = create_filter(T, grid, cutoff, filter_type = filter_type)
            prefactor = T(1 / down_factor^D)

            function CNOdownsampler(x)
                x_filter = filter(x) * prefactor
                x_filter[filtered_size...]
            end
        end
        direct_ds = direct_downsampler(T, D, N, down_factor, cutoff)
        @test direct_ds(u) ≈ ds(u)
    end

    @testset "Downsample Kernel AD" begin
        x_filter = CUDA.rand(Float32, 16, 16, 2, 1)
        result = CUDA.zeros(Float32, 8, 8, 2, 1)
        down_factor = 2
        mydev = Dict("bck" => CUDABackend(), "workgroupsize" => 256, "T" => Float32)
        downsample_kernel(mydev, x_filter, down_factor, 16)
        @test sum(result) !== 0.0

        y, back = Zygote.pullback(downsample_kernel, mydev, x_filter, 16, down_factor)
        x_filter_bar = CUDA.zeros(Float32, size(x_filter))
        result_bar = CUDA.rand(Float32, size(result))
        _, x_filter_bar, _, _ = back(result_bar)
        filtered_size = (((1:down_factor:16) for _ = 1:D)..., :, :)
        redown = x_filter_bar[filtered_size...]
        @test redown == result_bar

    end
    @testset "Downsample AD" begin
        u_test = CUDA.ones(Float32, size(u))
        du = ds(u_test)
        y, back = Zygote._pullback(ds, u_test)
        _, du_bar = back(du)
        @test size(du_bar) == size(u_test)
        @test sum(du_bar) !== 0.0
    end

    @testset "Upsample AD" begin
        u_test = CUDA.ones(Float32, size(u))
        du = us(u_test)
        y, back = Zygote._pullback(us, u_test)
        _, du_bar = back(du)
        @test size(du_bar) == size(u_test)
        @test sum(du_bar) !== 0.0
    end

    @testset "Down->Up AD" begin
        layers = (x -> ds(x), x -> us2(x))
        closure = Lux.Chain(layers...)
        rng = Random.Xoshiro(123)
        θ, st = Lux.setup(rng, closure)
        θ = ComponentArray(θ)
        function downup(u)
            closure(u, θ, st)[1]
        end
        u_test = CUDA.ones(Float32, size(u))
        du = downup(u_test)
        @test size(du) == size(u_test)
        y, back = Zygote._pullback(downup, u_test)
        _, du_bar = back(du)
        @test size(du_bar) == size(u_test)
        @test sum(du_bar) !== 0.0

    end
end
