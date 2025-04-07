# ConvolutionalNeuralOperators

[![Stable Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://DEEPDIP-project.github.io/ConvolutionalNeuralOperators.jl/stable)
[![In development documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://DEEPDIP-project.github.io/ConvolutionalNeuralOperators.jl/dev)
[![Build Status](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/workflows/Test/badge.svg)](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions)
[![Test workflow status](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions/workflows/Test.yml/badge.svg?branch=main)](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions/workflows/Test.yml?query=branch%3Amain)
[![Lint workflow Status](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions/workflows/Lint.yml/badge.svg?branch=main)](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions/workflows/Lint.yml?query=branch%3Amain)
[![Docs workflow Status](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions/workflows/Docs.yml/badge.svg?branch=main)](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/actions/workflows/Docs.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/DEEPDIP-project/ConvolutionalNeuralOperators.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DEEPDIP-project/ConvolutionalNeuralOperators.jl)
[![DOI](https://zenodo.org/badge/887124272.svg)](https://doi.org/10.5281/zenodo.14191802)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![All Contributors](https://img.shields.io/github/all-contributors/DEEPDIP-project/ConvolutionalNeuralOperators.jl?labelColor=5e1ec7&color=c0ffee&style=flat-square)](#contributors)
[![BestieTemplate](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/JuliaBesties/BestieTemplate.jl/main/docs/src/assets/badge.json)](https://github.com/JuliaBesties/BestieTemplate.jl)

This package implements [Convolutional Neural Operators](https://medium.com/@bogdan.raonke/operator-learning-convolutional-neural-operators-for-robust-and-accurate-learning-of-pdes-ebbc43b57434) following [this](https://github.com/camlab-ethz/ConvolutionalNeuralOperator).
The CNOs can then be used as custom Lux models and they are compatible with [closure modeling](https://github.com/DEEPDIP-project/CoupledNODE.jl).

## Install

```julia
using Pkg
Pkg.add(url="git@github.com:DEEPDIP-project/ConvolutionalNeuralOperator.jl.git")
```

## Usage

You probably want to use the `cno` function to create a closure model, which can be used in CoupledNODE or as a Lux model.

```julia
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
```

to get the closure model, and then use it as a Lux model, or in CoupledNODE

```julia
  l, trainstate = CoupledNODE.train(
      closure,
      θ,
      st,
      dataloader,
      loss;
      tstate = trainstate,
      nepochs = 2,
      alg = Adam(T(1.0e-3),
  )
```

Look in `test/` for more detailed examples on how to use the package, or look at the documentation.

## How to Cite

If you use ConvolutionalNeuralOperators.jl in your work, please cite using the reference given in [CITATION.cff](https://github.com/DEEPDIP-project/ConvolutionalNeuralOperators.jl/blob/main/CITATION.cff).

## Contributing

If you want to make contributions of any kind, please first that a look into our [contributing guide directly on GitHub](docs/src/90-contributing.md) or the [contributing page on the website](https://DEEPDIP-project.github.io/ConvolutionalNeuralOperators.jl/dev/90-contributing/)

---

### Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SCiarella"><img src="https://avatars.githubusercontent.com/u/58949181?v=4?s=100" width="100px;" alt="SCiarella"/><br /><sub><b>SCiarella</b></sub></a><br /><a href="#code-SCiarella" title="Code">💻</a> <a href="#test-SCiarella" title="Tests">⚠️</a> <a href="#maintenance-SCiarella" title="Maintenance">🚧</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
