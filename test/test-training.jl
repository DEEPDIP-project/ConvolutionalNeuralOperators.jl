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
model = create_CNO(
    T = T,
    N = N,
    D = D,
    cutoff = cutoff,
    ch_sizes = ch_,
    activations = act,
    down_factors = df,
    k_radii = k_rad,
    bottleneck_depths = bd,
    force_cpu = true,
)
θ, st = Lux.setup(rng, model)
θ = ComponentArray(θ)


@testset "CNO Model Training (CPU)" begin


    @testset "Initial Image Dimensions" begin
        @test size(u0) == (N0, N0, D, 6)
        @test size(u) == (N, N, D, 6)
    end

    @testset "Model output" begin
        @test size(model(u, θ, st)[1]) == size(u)
    end


    #    @testset "Loss" begin
    #        u_in = rand(T, size(u))
    #        tgt = rand(T, size(u))
    #
    #        # Define loss function
    #        function loss(θ, batch = 16)
    #            yout = model(u_in, θ, st)[1]
    #            return sum(abs2, (yout .- tgt))
    #        end
    #
    #        loss_0 = loss(θ, 128)
    #        @info "Initial loss is $(loss_0)"
    #        @test isfinite(loss_0)  # Ensure initial loss is a finite number
    #
    #        y, back = Zygote.pullback(loss, θ)
    #        @test y ≈ loss(θ)  # Ensure pullback is correct
    #        y_bar = rand(T, size(y))
    #        θ_bar = back(y_bar)[1]
    #        @test sum(θ_bar) !== 0.0  # Ensure gradient is non-zero
    #
    #        #        # Gradient calculation
    #        #        g = Zygote.gradient(θ -> loss(θ), θ)
    #        #        @test !isnothing(g)  # Ensure gradient is calculated successfully
    #        #        @info "gradient is $(g)"
    #    end


    #    @testset "Training" begin
    #        loss_0 = loss(θ, 128)
    #        # Callback function for optimization
    #        function callback(p, l_train)
    #            println("Training Loss: $(l_train)")
    #            false
    #        end
    #        # Test optimization
    #        optf =
    #            Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
    #        optprob = Optimization.OptimizationProblem(optf, θ)
    #        ClipAdam = OptimiserChain(Adam(1.0e-1), ClipGrad(1))
    #        optim_result, optim_t, optim_mem, _ =
    #            @timed Optimization.solve(optprob, ClipAdam; maxiters = 10, callback = callback)
    #
    #        # Final loss test
    #        loss_final = loss(optim_result.u, 128)
    #        @test loss_final < loss_0  # Ensure loss decreases after optimization
    #    end

    #    @testset "Full CNO model" begin
    #        model, θ, st = cno(
    #            T = T,
    #            N = N,
    #            D = D,
    #            cutoff = cutoff,
    #            ch_sizes = ch_,
    #            activations = act,
    #            down_factors = df,
    #            k_radii = k_rad,
    #            bottleneck_depths = bd,
    #            rng = rng,
    #            use_cuda = false,
    #        )
    #
    #        @test size(model(u, θ, st)[1]) == size(u)
    #        function loss(θ, batch = 16)
    #            y = rand(T, N, N, 1, batch)
    #            y = cat(y, y, dims = 3)
    #            yout = model(y, θ, st)[1]
    #            return sum(abs2, (yout .- y))
    #        end
    #        loss_0 = loss(θ, 128)
    #        @test isfinite(loss_0)  # Ensure initial loss is a finite number
    #        g = Zygote.gradient(θ -> loss(θ), θ)
    #        @test !isnothing(g)  # Ensure gradient is calculated successfully
    #        function callback(p, l_train)
    #            println("Training Loss: $(l_train)")
    #            false
    #        end
    #        optf =
    #            Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
    #        optprob = Optimization.OptimizationProblem(optf, θ)
    #        ClipAdam = OptimiserChain(Adam(1.0e-1), ClipGrad(1))
    #        optim_result, optim_t, optim_mem, _ =
    #            @timed Optimization.solve(optprob, ClipAdam, maxiters = 10, callback = callback)
    #        loss_final = loss(optim_result.u, 128)
    #        @test loss_final < loss_0
    #    end
end
