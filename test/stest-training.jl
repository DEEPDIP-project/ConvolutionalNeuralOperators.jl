using DifferentialEquations: ODEProblem, solve
using Optimization: Optimization, solve
using OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
using Random: Random
using TestImages: testimage
using ComponentArrays: ComponentArray
using Lux: Lux
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO
using NNlib: tanh_fast
using Zygote: Zygote
using Plots: heatmap, plot
using Test  # Importing the Test module for @test statements

# Initialize test suite
@testset "CNO Model Training" begin

    rng = Random.Xoshiro(123)
    N0 = 512
    D = 2
    T = Float32
    u0 = zeros(T, N0, N0, D, 6)

    # Load images into array and check sizes
    img_list = ["cameraman", "brick_wall_512", "fabio_gray_512", "livingroom", "pirate"]
    for i = 1:5
        img_name = img_list[i]
        u0[:, :, 1, i] .= testimage(img_name)
        u0[:, :, 2, i] .= testimage(img_name)
    end

    # Test the initial array setup
    @test size(u0) == (N0, N0, D, 6)

    # Downsize the input
    down_factor = 4
    cutoff = 0.1
    ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
    u = ds(u0)
    N = size(u)[1]

    # Test the downsampled image
    @test size(u) == (N, N, D, 6)

    # Model configuration
    ch_ = [2, 2]
    act = [tanh_fast, identity]
    df = [2, 2]
    k_rad = [3, 3]
    bd = [2, 2, 2]
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
    )
    θ, st = Lux.setup(rng, model)
    θ = ComponentArray(θ)

    # Test that model output has the same size as input
    @test size(model(u, θ, st)[1]) == size(u)

    # Define loss function
    function loss(θ, batch = 16)
        y = rand(T, N, N, 1, batch)
        y = cat(y, y, dims = 3)
        yout = model(y, θ, st)[1]
        return sum(abs2, (yout .- y))
    end

    # Initial loss test
    loss_0 = loss(θ, 128)
    @test isfinite(loss_0)  # Ensure initial loss is a finite number

    # Gradient calculation
    g = Zygote.gradient(θ -> loss(θ), θ)
    @test !isnothing(g)  # Ensure gradient is calculated successfully

    # Callback function for optimization
    function callback(p, l_train)
        println("Training Loss: $(l_train)")
        false
    end

    # Test optimization
    optf = Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, θ)
    ClipAdam = OptimiserChain(Adam(1.0e-1), ClipGrad(1))
    optim_result, optim_t, optim_mem, _ =
        @timed Optimization.solve(optprob, ClipAdam; maxiters = 10, callback = callback)

    # Final loss test
    loss_final = loss(optim_result.u, 128)
    @test loss_final < loss_0  # Ensure loss decreases after optimization

    # Now test the CUDA wrapper
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

    @info "There are $(length(θ)) parameters"

    @test size(model(u, θ, st)[1]) == size(u)
    function loss(θ, batch = 16)
        y = rand(T, N, N, 1, batch)
        y = cat(y, y, dims = 3)
        yout = model(y, θ, st)[1]
        return sum(abs2, (yout .- y))
    end
    loss_0 = loss(θ, 128)
    @test isfinite(loss_0)  # Ensure initial loss is a finite number
    g = Zygote.gradient(θ -> loss(θ), θ)
    @test !isnothing(g)  # Ensure gradient is calculated successfully
    function callback(p, l_train)
        println("Training Loss: $(l_train)")
        false
    end
    optf = Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
    optprob = Optimization.OptimizationProblem(optf, θ)
    ClipAdam = OptimiserChain(Adam(1.0e-1), ClipGrad(1))
    optim_result, optim_t, optim_mem, _ =
        @timed Optimization.solve(optprob, ClipAdam, maxiters = 10, callback = callback)
    loss_final = loss(optim_result.u, 128)
    @test loss_final < loss_0
end
