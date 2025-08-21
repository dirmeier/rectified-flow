# Rectified flows

[![ci](https://github.com/dirmeier/rectified-flow/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/rectified-flow/actions/workflows/ci.yaml)

## About

This repository implements a rectified flow
which has been proposed in [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
using JAX and Flax.

## Example usage

The `experiments` folder contains a use case where samples from the "Wto Moonss" data set are transported
to the "Eight Gaussian" data set. To run the example, first download the latest release
and install all dependencies via:

```bash
wget -qO- https://github.com/dirmeier/rectified-flow/archive/refs/tags/<TAG>.tar.gz | tar zxvf -
uv sync --all-groups
```

To train a model and make visualizations, call:

```bash
cd experiments/eight_gaussians_two_moons
python main.py
```

Shown below are samples from the two moons data set (black) that have been transported
to the eight Gaussians data set(blue). Each figure shows the transport
map after x training iterations.

<div align="center">
  <img src="experiments/eight_gaussians_two_moons/figures/samples-1.png" height="175">
    <img src="experiments/eight_gaussians_two_moons/figures/samples-1000.png" height="175">
<img src="experiments/eight_gaussians_two_moons/figures/samples-2000.png" height="175">
    <img src="experiments/eight_gaussians_two_moons/figures/samples-3000.png" height="175">
<img src="experiments/eight_gaussians_two_moons/figures/samples-4000.png" height="175">
    <img src="experiments/eight_gaussians_two_moons/figures/samples-10000.png" height="175">
</div>


## Installation

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/dirmeier/rflow@<TAG>
```

## Author

Simon Dirmeier <a href="mailto:simd23@pm.me">simd23 @ pm dot me</a>
