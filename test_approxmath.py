import approxmath as npa
import math
import numpy as np
import pytest

posxs = np.arange(0.01, 10.0, 0.01)
xs = np.arange(-10.0, 10.0, 0.01)

def test_battery():
    assert np.all(np.isclose(npa.log(posxs), np.log(posxs)))
    assert np.all(np.isclose(npa.exp(xs), np.exp(xs)))
    assert np.all(np.isclose(npa.sin(xs), np.sin(xs)))
    assert np.all(np.isclose(npa.cos(xs), np.cos(xs)))

@pytest.mark.benchmark(group="log")
def test_fast_log_bench(benchmark):
    benchmark(npa.log, posxs)

@pytest.mark.benchmark(group="log")
def test_np_log_bench(benchmark):
    benchmark(np.log, posxs)

@pytest.mark.benchmark(group="exp")
def test_fast_exp_bench(benchmark):
    benchmark(npa.exp, xs)

@pytest.mark.benchmark(group="exp")
def test_np_exp_bench(benchmark):
    benchmark(np.exp, xs)

@pytest.mark.benchmark(group="cos")
def test_fast_cos_bench(benchmark):
    benchmark(npa.cos, xs)

@pytest.mark.benchmark(group="cos")
def test_np_cos_bench(benchmark):
    benchmark(np.cos, xs)

@pytest.mark.benchmark(group="sin")
def test_fast_sin_bench(benchmark):
    benchmark(npa.sin, xs)

@pytest.mark.benchmark(group="sin")
def test_np_sin_bench(benchmark):
    benchmark(np.sin, xs)
