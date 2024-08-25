# Rectified flows

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/rectified-flow/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/rectified-flow/actions/workflows/ci.yaml)

> A rectified flow implementation in Flax

## About

This repository implements a rectified flow
which has been proposed in [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)
using JAX and Flax.

## Example usage

The `experiments` folder contains a use case where samples from the "Wto Moonss" data set are transported
to the "Eight Gaussian" data set. To train a model and make visualizations, call:

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
pip install git+https://github.com/dirmeier/rflow@<RELEASE>
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
