using DifferentialEquations: ODEProblem, solve, Tsit5
using Optimization: Optimization
using OptimizationOptimisers: OptimiserChain, Adam, ClipGrad
using Random: Random
using TestImages: testimage
using ComponentArrays: ComponentArray
using Lux: Lux
using ConvolutionalNeuralOperator: create_CNOdownsampler, create_CNO
using NNlib: tanh_fast
using Zygote: Zygote

rng = Random.Xoshiro(123)
N0 = 512
D = 2
T = Float32
u0 = zeros(T, N0, N0, D, 6)
u0[:, :, 1, 1] = testimage("cameraman")
u0[:, :, 2, 1] = testimage("cameraman")
u0[:, :, 1, 2] = testimage("brick_wall_512")
u0[:, :, 2, 2] = testimage("brick_wall_512")
u0[:, :, 1, 3] = testimage("fabio_gray_512")
u0[:, :, 2, 3] = testimage("fabio_gray_512")
u0[:, :, 1, 4] = testimage("lena_gray_512")
u0[:, :, 2, 4] = testimage("lena_gray_512")
u0[:, :, 1, 5] = testimage("livingroom")
u0[:, :, 2, 5] = testimage("livingroom")
u0[:, :, 1, 6] = testimage("pirate")
u0[:, :, 2, 6] = testimage("pirate")

# downsize the input which would be too large to autodiff
down_factor = 4
cutoff = 0.1
ds = create_CNOdownsampler(T, D, N0, down_factor, cutoff)
u = ds(u0)
N = size(u, 1)
heatmap(u[:, :, 2, 3], aspect_ratio = 1, title = "downsampled")

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
);
θ, st = Lux.setup(rng, model);
using ComponentArrays: ComponentArray
θ = ComponentArray(θ)


@assert size(model(u, θ, st)[1]) == size(u)
#heatmap(model(u, θ, st)[1][:, :, 1, 4], aspect_ratio = 1, title = "model(u0)")

model(u, θ, st)[1] .- u
function loss(θ)
    û = model(u, θ, st)[1]
    return sum(abs2, (û .- u) ./ u)
end
function loss(θ, batch = 16)
    y = rand(T, N, N, 1, batch)
    y = cat(y, y, dims = 3)
    yout = model(y, θ, st)[1]
    return sum(abs2, (yout .- y))
end
loss(θ)
g = Zygote.gradient(θ -> loss(θ), θ)

function callback(p, l_train)
    @info "Training Loss: $(l_train)"
    println(p.u[1])
    false
end

# test training with optimize
optf = Optimization.OptimizationFunction((p, _) -> loss(p), Optimization.AutoZygote())
optprob = Optimization.OptimizationProblem(optf, θ)
ClipAdam = OptimiserChain(Adam(1.0e-1), ClipGrad(1));
optim_result, optim_t, optim_mem, _ = @timed Optimization.solve(
    optprob,
    ClipAdam,
    maxiters = 30,
    callback = callback,
    progress = true,
)
θ_p = optim_result.u
#p1 = heatmap(model(u, θ_p, st)[1][:, :, 1, 1], aspect_ratio = 1, title = "model(u0)")
#p2 = heatmap(u[:, :, 1, 1], aspect_ratio = 1, title = "u0")
#p3 = heatmap(model(u, θ_p, st)[1][:, :, 1, 3], aspect_ratio = 1, title = "model(u0)")
#p4 = heatmap(u[:, :, 1, 3], aspect_ratio = 1, title = "u0")
#plot(p1, p2, p3, p4)

θ = θ_p

heatmap(
    (θ.down_k[1, :, :].*st.masks_down[1][:, :, 1, 1])[
        1:(2*model.k_radii[1]+1+1),
        1:(2*model.k_radii[1]+1+1),
    ],
    aspect_ratio = 1,
    title = "masks_down",
)
heatmap(
    (θ.down_k[4, :, :].*st.masks_down[2][:, :, 1, 1])[
        1:(2*model.k_radii[2]+1+1),
        1:(2*model.k_radii[2]+1+1),
    ],
    aspect_ratio = 1,
    title = "masks_down",
)