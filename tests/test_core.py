"""Checks of the core functions on synthetic data (no download needed):  pytest tests"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dmk_mwe import transport as tr, solvation as sv, shells as sh, raman as ra

RNG = np.random.default_rng(1)


def msd_direct(x):
    T = len(x)
    return np.array([((x[m:] - x[:T - m]) ** 2).sum(-1).mean() for m in range(T)])


def test_msd_fft_matches_direct_average():
    x = np.cumsum(RNG.normal(size=(300, 4, 3)), axis=0)
    assert np.allclose(tr.msd_fft(x), msd_direct(x))


def test_random_walk_diffusion():
    x = np.cumsum(RNG.normal(scale=0.1, size=(4000, 200, 3)), axis=0)     # D = 0.1^2 / 2 per unit time
    t = np.arange(4000.0)
    msd = tr.msd_fft(x)
    assert abs(tr.slope(t, msd, (100, 1000)) / 6 - 0.005) < 3e-4


def test_onsager_terms_sum_to_collective_msd():
    cat = np.cumsum(RNG.normal(size=(500, 5, 3)), axis=0)
    an = np.cumsum(RNG.normal(size=(500, 5, 3)), axis=0)
    o, N = tr.onsager_msds(cat, an), 5
    m = tr.msd_fft(cat.sum(1) - an.sum(1))                                # charge displacement, q = +-1
    rebuilt = (N * o["self_cation"] + N * o["self_anion"] + N * (N - 1) * (o["cat_cat"] + o["an_an"])
               - 2 * N * N * o["cat_an"])
    assert np.allclose(rebuilt, m)


def test_ideal_gas_rdf_is_one():
    L, X = 20.0, RNG.uniform(0, 20.0, size=(200, 120, 3))
    r, g = sv.rdf(X, np.arange(60), np.arange(60, 120), L)
    k = (r > 3) & (r < 9)
    assert abs(g[k].mean() - 1) < 0.02
    rho = 60 / L ** 3
    assert abs(sv.cn_at(r, np.ones_like(r), rho, 5.0) - 4 / 3 * np.pi * 125 * rho) < 0.02


def test_speciation_counts_each_anion_once():
    # 2 frames, 3 anions: frame 0 anion 0 bound to Li 0 twice (two O), anion 1 bound to Li 0 and Li 1
    frame, anion, li = np.array([0, 0, 0, 0, 1]), np.array([0, 0, 1, 1, 2]), np.array([0, 0, 0, 1, 3])
    assert np.allclose(sv.speciation(frame, anion, li, 2, 3), 100 * np.array([3, 2, 1]) / 6)


def test_residence_time_of_exponential():
    C = np.exp(-np.arange(400) / 25.0)
    assert abs(sh.residence_time(C, 5.0) - 125.0) < 1e-3


def test_acf_of_a_single_long_bond():
    T = 200
    key, frame = np.zeros(100, int), np.arange(100)                       # bound for the first half only
    C = sh.acf(key, frame, T)
    assert C[0] == 1 and abs(C[50] - 50 / 150 / (100 / 200)) < 1e-6


def test_dG_five():
    P = np.array([0, 0, 0, 0, 0.5, 0.5, 0, 0])
    assert abs(sh.dG_five(P)) < 1e-12


def test_raman_n_limits():
    assert abs(ra.raman_n(4.0, r=1.0) - 4.0) < 1e-12                      # equal cross sections: N = CN
    assert ra.raman_n(4.0, r=0.86) < 4.0


def test_band_fit_recovers_two_bands():
    X = ra.GRID
    y = ra.lg(X, 1.0, 789.0, 5.8, 20.0) + ra.lg(X, 0.6, 803.0, 5.8, 20.0) + 0.01
    f = ra.fit_band(y / y.max(), "common")
    assert abs(f["bound_fraction"] - 0.375) < 0.01
