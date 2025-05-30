using Test
using Adapt
using Lux
using LuxCUDA
using JLD2
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO
using ComponentArrays: ComponentArray
using Optimisers: Adam, ClipGrad, OptimiserChain
using Random
using Zygote: Zygote
using CUDA
using CoupledNODE
using IncompressibleNavierStokes
using NeuralClosure
using OrdinaryDiffEqTsit5

rng = Random.Xoshiro(123)
T = Float32
N = 16
nles = 16
D = 2
ch_ = [2, 2]
act = [tanh_fast, identity]
df = [2, 2]
k_rad = [3, 3]
bd = [2, 2, 2]
cutoff = 10
batch = 4

@testset "CoupledNODE integration (CPU)" begin
    # Create the model
    closure, θ_start, st = cno(
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

    # Define input tensor and pass through model
    input_tensor = rand(T, N, N, D, batch)
    output = Lux.apply(closure, input_tensor, θ_start, st)[1]
    @test size(output) == size(input_tensor)

    # Read conf
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    conf = NS.read_config("./config.yaml")
    conf["params"]["backend"] = IncompressibleNavierStokes.CPU()

    # get params
    params = NS.load_params(conf)
    device(x) = x

    # Get the setup in the format expected by the CoupledNODE
    function getsetup(; params, nles)
        Setup(;
            x = ntuple(α -> range(params.lims..., nles + 1), params.D),
            params.Re,
            params.backend,
            params.bodyforce,
            params.issteadybodyforce,
        )
    end
    setup = getsetup(; params, nles)
    psolver = default_psolver(setup)
    setup = []
    for nl in nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
        push!(setup, Setup(; x = x, Re = params.Re, params.backend))
    end

    # Load data
    function namedtupleload(file)
        dict = load(file)
        k, v = keys(dict), values(dict)
        pairs = @. Symbol(k) => v
        (; pairs...)
    end
    data_train = load("data_train.jld2", "data_train")

    # Create the io array
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_posteriori(data_train, setup[1])

    # Create the dataloader
    θ = device(copy(θ_start))
    nunroll = 2
    nunroll_valid = 2
    dataloader_post = NS.create_dataloader_posteriori(
        io_train;
        nunroll = nunroll,
        rng = Random.Xoshiro(24),
    )
    u, t = dataloader_post()

    # Create the right hand side and the loss
    dudt_nn = NS.create_right_hand_side_with_closure(setup[1], psolver, closure, st)
    griddims = ((:) for _ = 1:D)
    loss = CoupledNODE.create_loss_post_lux(
        dudt_nn,
        griddims,
        griddims;
        sciml_solver = Tsit5(),
        force_cpu = true,
    )
    callbackstate = trainstate = nothing


    # For testing reason, explicitely set up the probelm
    # Notice that this is automatically done in CoupledNODE
    u, t = dataloader_post()
    x = u[griddims..., :, 1, 1]
    y = u[griddims..., :, 1, 2:end] # remember to discard sol at the initial time step
    tspan, dt, prob, pred = nothing, nothing, nothing, nothing # initialize variable outside allowscalar do.
    dt = @views t[2:2] .- t[1:1]
    dt = only(Array(dt))
    function get_tspan(t)
        return (Array(t)[1], Array(t)[end])
    end
    tspan = get_tspan(t)
    prob = ODEProblem(dudt_nn, x, tspan, θ)
    pred = Array(solve(prob, Tsit5(); u0 = x, p = θ, adaptive = true, saveat = Array(t)))

    # Test the forward pass
    @test size(pred[:, :, :, 2:end]) == size(y)


    # Test the backward pass
    p = prob.p
    y = prob.u0
    f = prob.f
    λ = zero(prob.u0)
    _dy, back = Zygote.pullback(y, p) do u, p
        vec(f(u, p, t))
    end
    tmp1, tmp2 = back(λ)
    @test size(tmp1) == (18, 18, 2)
    @test size(tmp2) == (6656,)

    # Final integration test of the entire train interface
    l, trainstate = CoupledNODE.train(
        closure,
        θ,
        st,
        dataloader_post,
        loss;
        tstate = trainstate,
        nepochs = 2,
        alg = OptimiserChain(Adam(T(1.0e-3)), ClipGrad(0.1)),
        cpu = true,
    )
    @test isnan(l) == false
    @test trainstate.step == 2
    @test any(isnan, trainstate.parameters) == false

end


@testset "CoupledNODE integration (GPU)" begin
    if !CUDA.functional()
        @warn "CUDA not functional, skipping GPU tests"
        return
    end
    CUDA.allowscalar(false)

    # Create the model
    closure, θ_start, st = cno(
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

    # Define input tensor and pass through model
    input_tensor = CUDA.rand(T, N, N, D, batch)
    output = Lux.apply(closure, input_tensor, θ_start, st)[1]
    @test size(output) == size(input_tensor)

    # Read conf
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    conf = NS.read_config("./config.yaml")
    conf["params"]["backend"] = CUDABackend()

    # get params
    params = NS.load_params(conf)
    device(x) = adapt(params.backend, x)

    # Get the setup in the format expected by the CoupledNODE
    function getsetup(; params, nles)
        Setup(;
            x = ntuple(α -> range(params.lims..., nles + 1), params.D),
            params.Re,
            params.backend,
            params.bodyforce,
            params.issteadybodyforce,
        )
    end
    setup = getsetup(; params, nles)
    psolver = default_psolver(setup)
    setup = []
    for nl in nles
        x = ntuple(α -> LinRange(T(0.0), T(1.0), nl + 1), params.D)
        push!(setup, Setup(; x = x, Re = params.Re, params.backend))
    end

    # Load data
    function namedtupleload(file)
        dict = load(file)
        k, v = keys(dict), values(dict)
        pairs = @. Symbol(k) => v
        (; pairs...)
    end
    data_train = load("data_train.jld2", "data_train")

    # Create the io array
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_posteriori(data_train, setup[1], device)

    # Create the dataloader
    θ = device(copy(θ_start))
    nunroll = 2
    nunroll_valid = 2
    dataloader_post = NS.create_dataloader_posteriori(
        io_train;
        nunroll = nunroll,
        rng = Random.Xoshiro(24),
        device = device,
    )
    u, t = dataloader_post()

    # Create the right hand side and the loss
    dudt_nn = NS.create_right_hand_side_with_closure(setup[1], psolver, closure, st)
    griddims = ((:) for _ = 1:D)
    loss = CoupledNODE.create_loss_post_lux(
        dudt_nn,
        griddims,
        griddims;
        sciml_solver = Tsit5(),
    )
    callbackstate = trainstate = nothing


    # For testing reason, explicitely set up the probelm
    # Notice that this is automatically done in CoupledNODE
    x, y = nothing, nothing
    CUDA.allowscalar() do
        x = u[griddims..., :, 1, 1]
        y = u[griddims..., :, 1, 2:end] # remember to discard sol at the initial time step
    end
    tspan, dt, prob, pred = nothing, nothing, nothing, nothing # initialize variable outside allowscalar do.
    function get_tspan(t)
        return (Array(t)[1], Array(t)[end])
    end
    tspan = get_tspan(t)
    prob = ODEProblem(dudt_nn, x, tspan, θ)
    pred = Array(solve(prob, Tsit5(); u0 = x, p = θ, adaptive = true, saveat = Array(t)))

    # Test the forward pass
    @test size(pred[:, :, :, 2:end]) == size(y)


    # Test the backward pass
    p = prob.p
    y = prob.u0
    f = prob.f
    λ = CUDA.zero(prob.u0)
    _dy, back = Zygote.pullback(y, p) do u, p
        vec(f(u, p, t))
    end
    tmp1, tmp2 = back(λ)
    @test size(tmp1) == (18, 18, 2)
    @test size(tmp2) == (6656,)
    @test isa(tmp1, CUDA.CuArray)

    # Final integration test of the entire train interface
    l, trainstate = CoupledNODE.train(
        closure,
        θ,
        st,
        dataloader_post,
        loss;
        tstate = trainstate,
        nepochs = 2,
        alg = OptimiserChain(Adam(T(1.0e-3)), ClipGrad(0.1)),
    )
    @test isnan(l) == false
    @test trainstate.step == 2
    @test any(isnan, trainstate.parameters) == false

end
