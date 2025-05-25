using DifferentialEquations: ODEProblem, solve
using Optimization: Optimization, solve
using OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
using Random: Random
using TestImages: testimage
using ComponentArrays: ComponentArray
using Lux: Lux
using CUDA
using LuxCUDA
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO
using NNlib: tanh_fast
using Zygote: Zygote
using Test  # Importing the Test module for @test statements
using ChainRulesCore: @ignore_derivatives

rng = Random.Xoshiro(123)
N0 = 512
D = 2
T = Float32
u0 = zeros(T, N0, N0, D, 6)
img_list = ["cameraman", "brick_wall_512", "fabio_gray_512", "livingroom", "pirate"]
for i = 1:5
    img_name = img_list[i]
    u0[:, :, 1, i] .= testimage(img_name)
    u0[:, :, 2, i] .= testimage(img_name)
end
# Downsize the input
down_factor0 = 8
cutoff = 0.1
ds = create_CNOdownsampler(T, D, N0, down_factor0, cutoff, force_cpu = true)
u = ds(u0)
N = size(u)[1]
# Model configuration
ch_ = [2]
act = [tanh_fast]
df = [2]
k_rad = [2]
bd = [2, 2]
cutoff = 10


@testset "Full model (CPU)" begin

    @testset "Full CNO model" begin
        model, θ, st = cno(
            T = T,
            N = N,
            D = D,
            cutoff = cutoff,
            ch_sizes = ch_,
            activations = act,
            down_factors = df,
            k_radii = k_rad,
            bottleneck_depths = bd,
            rng = rng,
            use_cuda = false,
        )

        @test size(model(u, θ, st)[1]) == size(u)

        u_in = rand(T, size(u))
        tgt = rand(T, size(u))
        function loss(θ, batch = 16)
            yout = model(u_in, θ, st)[1]
            return sum(abs2, (yout .- tgt))
        end
        loss_0 = loss(θ, 128)
        @test isfinite(loss_0)  # Ensure initial loss is a finite number

        g = Zygote.gradient(θ -> loss(θ), θ)
        @test !isnothing(g)  # Ensure gradient is calculated successfully
        function callback(p, l_train)
            println("Training Loss (wrapped model): $(l_train)")
            false
        end
        optf =
            Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optf, θ)
        ClipAdam = OptimiserChain(Adam(1.0e-1), ClipGrad(1))
        optim_result, optim_t, optim_mem, _ =
            @timed Optimization.solve(optprob, ClipAdam, maxiters = 10, callback = callback)
        loss_final = loss(optim_result.u, 128)
        @test loss_final < loss_0
    end
end

@testset "Full model (GPU)" begin
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    CUDA.allowscalar(false)

    @testset "Full CNO model" begin
        model, θ, st = cno(
            T = T,
            N = N,
            D = D,
            cutoff = cutoff,
            ch_sizes = ch_,
            activations = act,
            down_factors = df,
            k_radii = k_rad,
            bottleneck_depths = bd,
            rng = rng,
            use_cuda = true,
        )

        u_gpu = CuArray(u)
        y, _ = model(u_gpu, θ, st)
        @test size(y) == size(u)
        @test isa(y, CuArray)


        u_in = rand(T, size(u))
        tgt = rand(T, size(u))
        function loss(θ, batch = 16)
            yout = model(u_in, θ, st)[1]
            return sum(abs2, (yout .- tgt))
        end
        loss_0 = loss(θ, 128)
        @test isfinite(loss_0)  # Ensure initial loss is a finite number

        g = Zygote.gradient(θ -> loss(θ), θ)
        @test !isnothing(g)  # Ensure gradient is calculated successfully
        function callback(p, l_train)
            println("Training Loss (wrapped model) [GPU]: $(l_train)")
            false
        end
        optf =
            Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
        optprob = Optimization.OptimizationProblem(optf, θ)
        ClipAdam = OptimiserChain(Adam(1.0e-2), ClipGrad(1))
        optim_result, optim_t, optim_mem, _ =
            @timed Optimization.solve(optprob, ClipAdam, maxiters = 10, callback = callback)
        loss_final = loss(optim_result.u, 128)
        @test loss_final < loss_0
    end
end
