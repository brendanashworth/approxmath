import approxmath.np as npa
import approxmath.aesara as att
import aesara.tensor as tt
import aesara
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

x_val = np.random.rand(25, 30)

def test_aesara_log_op():
    x = tt.matrix()
    f = aesara.function([x], att.ApproxLogOp()(x))
    out = f(x_val)
    assert np.allclose(np.log(x_val), out)

def test_aesara_log():
    out = att.log(x_val)
    assert np.allclose(np.log(x_val), out)

def test_aesara_exp_op():
    x = tt.matrix()
    f = aesara.function([x], att.ApproxExpOp()(x))
    out = f(x_val)
    assert np.allclose(np.exp(x_val), out)

def test_aesara_exp():
    out = att.exp(x_val)
    assert np.allclose(np.exp(x_val), out)

def test_aesara_sin_op():
    x = tt.matrix()
    f = aesara.function([x], att.ApproxSinOp()(x))
    out = f(x_val)
    assert np.allclose(np.sin(x_val), out)

def test_aesara_sin():
    out = att.sin(x_val)
    assert np.allclose(np.sin(x_val), out)

def test_aesara_cos_op():
    x = tt.matrix()
    f = aesara.function([x], att.ApproxCosOp()(x))
    out = f(x_val)
    assert np.allclose(np.cos(x_val), out)

def test_aesara_cos():
    out = att.cos(x_val)
    assert np.allclose(np.cos(x_val), out)

def test_aesara_exp_grad():
    aesara.gradient.verify_grad(att.ApproxExpOp(), [x_val], rng=np.random.RandomState())

def test_aesara_log_grad():
    aesara.gradient.verify_grad(att.ApproxLogOp(), [x_val], rng=np.random.RandomState())

def test_aesara_sin_grad():
    aesara.gradient.verify_grad(att.ApproxSinOp(), [x_val], rng=np.random.RandomState())

def test_aesara_cos_grad():
    aesara.gradient.verify_grad(att.ApproxCosOp(), [x_val], rng=np.random.RandomState())

x_val_bench = np.random.rand(50, 50)

@pytest.mark.benchmark(group="aesara_log")
def test_att_log_bench(benchmark):
    benchmark(att.log, x_val_bench)

@pytest.mark.benchmark(group="aesara_log")
def test_tt_log_bench(benchmark):
    benchmark(tt.log, x_val_bench)

@pytest.mark.benchmark(group="aesara_exp")
def test_att_exp_bench(benchmark):
    benchmark(att.exp, x_val_bench)

@pytest.mark.benchmark(group="aesara_exp")
def test_tt_exp_bench(benchmark):
    benchmark(tt.exp, x_val_bench)

@pytest.mark.benchmark(group="aesara_sin")
def test_att_sin_bench(benchmark):
    benchmark(att.sin, x_val_bench)

@pytest.mark.benchmark(group="aesara_sin")
def test_tt_sin_bench(benchmark):
    benchmark(tt.sin, x_val_bench)

@pytest.mark.benchmark(group="aesara_cos")
def test_att_cos_bench(benchmark):
    benchmark(att.cos, x_val_bench)

@pytest.mark.benchmark(group="aesara_cos")
def test_tt_cos_bench(benchmark):
    benchmark(tt.cos, x_val_bench)

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
