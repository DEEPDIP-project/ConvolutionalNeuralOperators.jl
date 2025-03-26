using Adapt
using Random: Random
using ConvolutionalNeuralOperators: apply_residual_blocks
using Lux
using LuxCUDA
using Zygote: Zygote
using ComponentArrays: ComponentArray
using CUDA

CUDA.allowscalar(false)

x = rand(Float32, 16, 16, 2, 1)
k_bottlenecks = rand(Float32, 100, 16, 16)
k_residual = ((1:2,), (10:11,))
mask = rand(Float32, 16, 16)

@testset "Residual block (CPU)" begin

    y = apply_residual_blocks(x, k_bottlenecks, k_residual, mask, identity)

    @testset "Forward" begin
        @test size(y) == size(x)
    end

    @testset "AD" begin
        y_bar = ones(Float32, size(y))
        y, back = Zygote.pullback(
            apply_residual_blocks,
            x,
            k_bottlenecks,
            k_residual,
            mask,
            identity,
        )
        x_bar, k_bar, kr_bar, mask_bar, act_bar = back(y_bar)
        @test size(x_bar) == size(x)
        @test size(k_bar) == size(k_bottlenecks)
        @test all(isnothing(i...) for i in kr_bar)
        @test isnothing(act_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar[1:2, :, :]) !== 0.0
        @test sum(k_bar[3:9, :, :]) == 0.0
        @test sum(k_bar[10:11, :, :]) !== 0.0
        @test sum(k_bar[12:end, :, :]) == 0.0
    end



end

if !CUDA.functional()
    @test "CUDA not functional, skipping GPU tests"
    return
end
# Prepare for GPU tests
x = CUDA.rand(Float32, 16, 16, 2, 1)
k_bottlenecks = CUDA.rand(Float32, 100, 16, 16)
k_residual = ((1:2,), (10:11,))
mask = CUDA.rand(Float32, 16, 16)

@testset "Residual block (GPU)" begin

    y = apply_residual_blocks(x, k_bottlenecks, k_residual, mask, identity)

    @testset "Forward" begin
        @test size(y) == size(x)
        @test isa(y, CuArray)
    end

    @testset "AD" begin
        y_bar = CUDA.ones(Float32, size(y))
        y, back = Zygote.pullback(
            apply_residual_blocks,
            x,
            k_bottlenecks,
            k_residual,
            mask,
            identity,
        )
        x_bar, k_bar, kr_bar, mask_bar, act_bar = back(y_bar)
        @test size(x_bar) == size(x)
        @test size(k_bar) == size(k_bottlenecks)
        @test all(isnothing(i...) for i in kr_bar)
        @test isnothing(act_bar)
        @test sum(x_bar) !== 0.0
        @test sum(k_bar[1:2, :, :]) !== 0.0
        @test sum(k_bar[3:9, :, :]) == 0.0
        @test sum(k_bar[10:11, :, :]) !== 0.0
        @test sum(k_bar[12:end, :, :]) == 0.0
        @test isa(x_bar, CuArray)
        @test isa(k_bar, CuArray)
        @test isa(y, CuArray)
    end



end
