using ConvolutionalNeuralOperators
using Documenter

DocMeta.setdocmeta!(
    ConvolutionalNeuralOperators,
    :DocTestSetup,
    :(using ConvolutionalNeuralOperators);
    recursive = true,
)

const page_rename = Dict("developer.md" => "Developer docs") # Without the numbers
const numbered_pages = [
    file for file in readdir(joinpath(@__DIR__, "src")) if
    file != "index.md" && splitext(file)[2] == ".md"
]

makedocs(;
    modules = [ConvolutionalNeuralOperators],
    authors = "SCiarella <simoneciarella@gmail.com>",
    repo = "https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/blob/{commit}{path}#{line}",
    sitename = "ConvolutionalNeuralOperators.jl",
    format = Documenter.HTML(;
        canonical = "https://DEEPDIP-project.github.io/ConvolutionalNeuralOperators.jl",
    ),
    pages = ["index.md"; numbered_pages],
)

deploydocs(; repo = "github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl")
