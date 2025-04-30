using Test
using Adapt
using Lux
using LuxCUDA
using JLD2
using ConvolutionalNeuralOperators: create_CNOdownsampler, create_CNO
using ComponentArrays: ComponentArray
using Optimisers: Adam, ClipGrad, OptimiserChain
using Optimization
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
    data_train = []
    data_i = namedtupleload("data_train.jld2")
    push!(data_train, hcat(data_i))

    # Create the io array
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_priori(data_train, setup)

    # Create the dataloader
    θ = device(copy(θ_start))
    dataloader_prior = NS.create_dataloader_prior(
        io_train[1];
        batchsize = 4,
        rng = Random.Xoshiro(24),
        device = device,
    )
    train_data_priori = dataloader_prior()

    l0 = CoupledNODE.loss_priori_lux(closure, θ, st, train_data_priori)[1]
    @test isnan(l0) == false
    loss = CoupledNODE.loss_priori_lux

    # Final integration test of the entire train interface
    l, trainstate = CoupledNODE.train(
        closure,
        θ,
        st,
        dataloader_prior,
        loss;
        nepochs = 20,
        alg = OptimiserChain(Adam(T(1.0e-3)), ClipGrad(0.1)),
        cpu = true,
    )
    @test isnan(l) == false
    @test l < l0
    @test trainstate.step == 20
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
    @test isa(output, CuArray)

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
    data_train = []
    data_i = namedtupleload("data_train.jld2")
    push!(data_train, hcat(data_i))

    # Create the io array
    NS = Base.get_extension(CoupledNODE, :NavierStokes)
    io_train = NS.create_io_arrays_priori(data_train, setup)

    # Create the dataloader
    θ = device(copy(θ_start))
    dataloader_prior = NS.create_dataloader_prior(
        io_train[1];
        batchsize = 4,
        rng = Random.Xoshiro(24),
        device = device,
    )
    train_data_priori = dataloader_prior()
    @test isa(train_data_priori[1], CuArray)
    @test isa(train_data_priori[2], CuArray)

    l0 = CoupledNODE.loss_priori_lux(closure, θ, st, train_data_priori)[1]
    @test isnan(l0) == false
    loss = CoupledNODE.loss_priori_lux

    function loss_pb(model, ps, st, (x, y), device = identity)
        y_pred, st_ = Lux.apply(model, x, ps, st)[1:2]
        return sum(abs2, y_pred - y) / sum(abs2, y)
    end
    y, back = Zygote.pullback(loss_pb, closure, θ, st, train_data_priori)
    @test y == l0
    y_bar = 1
    _, θ_bar, _, _ = back(y_bar)
    @test size(θ_bar) == size(θ)
    @test sum(θ_bar) !== 0.0


    tstate = Lux.Training.TrainState(closure, θ, st, Adam(T(1.0e-3))) |> Lux.gpu_device()
    data = dataloader_prior()
    _, l, _, tstate =
        Lux.Training.single_train_step!(Optimization.AutoZygote(), loss, data, tstate) |>
        Lux.gpu_device()
    @test isnan(l) == false
    @test l < 2 * l0
    @test tstate.step == 1

    # Final integration test of the entire train interface
    l, trainstate = CoupledNODE.train(
        closure,
        θ,
        st,
        dataloader_prior,
        loss;
        nepochs = 20,
        alg = Adam(T(1.0e-3)),
    )
    @test isnan(l) == false
    @test l < 2 * l0
    @test trainstate.step == 20
    @test any(isnan, trainstate.parameters) == false

end
