"""Raman: bound-to-free intensity ratio r of the C-C-C stretch of DMK from the ORCA cluster calculations, and the
decomposition of the measured band (Methods, 'Raman cross-section calculations'; Fig. 3; SI 'Re-interpretation of the
experimental spectra').

Conventions (as published):
  modes       from the ORCA .hess file: wavenumbers, normal modes and Raman activities S = 45 a'^2 + 7 g'^2 (A^4/amu)
                                                                                              -> `read_hess`
  DMK atoms   C, H and the O bonded to C; the number of ligands is the number of carbonyl O. A mode belongs to DMK when
              at least half of its mass-weighted displacement is on DMK atoms                 -> `modes`
  band        the modes between 750 and 840 cm-1 (the fitted range of the measured band); one C-C-C stretch per ligand
  intensity   S_k f(nu_k), f = (nu0 - nu)^4 / {nu [1 - exp(-hc nu / kT)]}, 532 nm, 298 K (Pikulova & Neese 2026, Eq. 3)
  r           band intensity per DMK, cluster / free DMK at the same level                    -> `compare`
  shift       activity-weighted band centre, cluster - free
  rock        the a1 CH3-C rocks between 1000 and 1150 cm-1 (the lower group of DMK modes; Deng & Irish's 1064 cm-1
              internal standard); double ratio = r / rock ratio
  N           area fraction x c_DMK/c_Li with the bound band weighted by 1/r: N = 9 r CN / (r CN + 9 - CN)
                                                                                              -> `raman_n`
  band fits   two components and a constant baseline on 750-840 cm-1 (0.5 cm-1 grid), areas of the analytic
              components over 600-1000 cm-1; N = R A_b / (A_b + A_f), R = 13.55 / c           -> `fit_band`
"""
import os
import numpy as np
from scipy.optimize import least_squares

BOHR = 0.529177210903
BAND, ROCK = (750.0, 840.0), (1000.0, 1150.0)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def frequency_factor(nu, lam=532.0, kT=207.22):
    """(nu0 - nu)^4 / {nu [1 - exp(-hc nu / kT)]}; kT/hc = 207.22 cm-1 at 298 K."""
    nu0 = 1e7 / lam
    return (nu0 - nu) ** 4 / (nu * (1.0 - np.exp(-nu / kT)))


def read_hess(path):
    """Symbols, masses, coordinates (A), wavenumbers (cm-1), Raman activities and normal modes of an ORCA .hess file."""
    L = open(path).read().split("\n")
    sec = {l.strip(): i for i, l in enumerate(L) if l.startswith("$")}
    i = sec["$atoms"]
    n = int(L[i + 1])
    at = [L[i + 2 + a].split() for a in range(n)]
    sym = [a[0] for a in at]
    mass = np.array([float(a[1]) for a in at])
    X = np.array([a[2:5] for a in at], float) * BOHR
    i = sec["$vibrational_frequencies"]
    m = int(L[i + 1])
    freq = np.array([float(L[i + 2 + k].split()[1]) for k in range(m)])
    i = sec["$raman_spectrum"]
    S = np.array([float(L[i + 2 + k].split()[1]) for k in range(m)])
    i = sec["$normal_modes"]
    nr, nc = map(int, L[i + 1].split())
    M = np.zeros((nr, nc))
    j = i + 2
    while j < len(L) and not L[j].startswith("$") and L[j].strip():
        cols = [int(c) for c in L[j].split()]
        for r in range(nr):
            M[r, cols] = [float(v) for v in L[j + 1 + r].split()[1:]]
        j += nr + 1
    return sym, mass, X, freq, S, M.T


def static_polarizability(out_path):
    """Isotropic static polarizability (au) printed in the ORCA .out file, or None."""
    pol = None
    if os.path.exists(out_path):
        for line in open(out_path):
            if "Isotropic polarizability" in line:
                pol = float(line.split()[-1])
    return pol


def modes(hess):
    """Band, rock and polarizability data of one calculation (cluster or free DMK)."""
    sym, mass, X, freq, S, M = read_hess(hess)
    C = [a for a, s in enumerate(sym) if s == "C"]
    oc = [a for a, s in enumerate(sym) if s == "O" and min(np.linalg.norm(X[a] - X[c]) for c in C) < 1.4]
    dmk = sorted(set(C) | set(oc) | {a for a, s in enumerate(sym) if s == "H"})
    vib = np.where(np.abs(freq) > 1e-6)[0]
    fr, S, x = freq[vib], S[vib], M[vib].reshape(len(vib), -1, 3)
    w = mass[None, :] * (x ** 2).sum(2)
    on_dmk = w[:, dmk].sum(1) / w.sum(1)
    nl = len(oc)
    band = [k for k in range(len(fr)) if BAND[0] < fr[k] < BAND[1]]
    rk = [k for k in range(len(fr)) if ROCK[0] < fr[k] < ROCK[1] and on_dmk[k] >= 0.5]
    rock = rk[:np.argmax(np.diff(fr[rk])) + 1] if len(rk) > nl else rk          # the a1 group, below the gap
    assert len(band) == nl and all(on_dmk[k] >= 0.5 for k in band), f"{hess}: one DMK stretch per ligand expected"
    return dict(n_dmk=nl, n_imag=int((fr < 0).sum()),
                I=(S[band] * frequency_factor(fr[band])).sum() / nl,
                I_rock=(S[rock] * frequency_factor(fr[rock])).sum() / nl,
                nu=(fr[band] * S[band]).sum() / S[band].sum(),
                pol=static_polarizability(os.path.splitext(hess)[0] + ".out"))


def compare(cluster, free):
    """r, rock ratio, double ratio and band shift of a cluster against free DMK."""
    r, rr = cluster["I"] / free["I"], cluster["I_rock"] / free["I_rock"]
    return dict(r=r, rock=rr, double=r / rr, shift=cluster["nu"] - free["nu"])


def raman_n(cn, r=0.86, c_ratio=9.0):
    """Solvation number the C-C-C band reports for a shell of `cn` bound DMK at c_DMK / c_Li = c_ratio."""
    return c_ratio * r * cn / (r * cn + c_ratio - cn)


# ------------------------------------------------------------------ decomposition of the measured band
GRID = np.arange(750.0, 840.01, 0.5)


def lg(x, a, p, wl, wg):
    """Lorentzian-Gaussian product (James & Mayes 1982)."""
    return a / (1 + ((x - p) / wl) ** 2) * np.exp(-np.log(2) * ((x - p) / wg) ** 2)


def pv(x, a, p, w, eta):
    """Pseudo-Voigt."""
    return a * (eta / (1 + ((x - p) / w) ** 2) + (1 - eta) * np.exp(-np.log(2) * ((x - p) / w) ** 2))


def _area(fun, q):
    xx = np.arange(600, 1000, 0.05)
    return _trapz(fun(xx, *q), xx)


def fit_band(y, model):
    """Fit the normalised band y (on GRID) with a free (lower) and a bound (upper) component and a constant baseline.

    model: 'common'  Lorentzian-Gaussian products with a common width and shape
           'lg'      Lorentzian-Gaussian products, all parameters independent
           'voigt'   pseudo-Voigts, all parameters independent
    Returns the bound area fraction, the two components, the baseline and the component parameters."""
    X, pk = GRID, y.max()
    if model == "common":
        f = lambda q: [lg(X, q[0], q[1], q[4], q[5]), lg(X, q[2], q[3], q[4], q[5])]
        q0 = [pk * .7, 790, pk * .4, 806, 6, 20, 0]
        lo, hi = [0, 780, 0, 795, 1, 2, -pk], [np.inf, 800, np.inf, 820, 30, 300, pk]
        par = lambda q: ([q[0], q[1], q[4], q[5]], [q[2], q[3], q[4], q[5]])
        fun = lg
    elif model == "lg":
        f = lambda q: [lg(X, *q[0:4]), lg(X, *q[4:8])]
        q0 = [pk * 0.7, 790.0, 6, 20, pk * 0.4, 806.0, 7, 20, 0]
        lo, hi = [0, 780, 1, 2, 0, 795, 1, 2, -pk], [np.inf, 800, 30, 300, np.inf, 820, 30, 300, pk]
        par = lambda q: (q[0:4], q[4:8])
        fun = lg
    elif model == "voigt":
        f = lambda q: [pv(X, *q[0:4]), pv(X, *q[4:8])]
        q0 = [pk * .7, 790, 6, .7, pk * .4, 806, 7, .7, 0]
        lo, hi = [0, 780, 1, 0, 0, 795, 1, 0, -pk], [np.inf, 800, 30, 1, np.inf, 820, 30, 1, pk]
        par = lambda q: (q[0:4], q[4:8])
        fun = pv
    else:
        raise ValueError(model)
    res = least_squares(lambda q: sum(f(q)) + q[-1] - y, q0, bounds=(lo, hi), max_nfev=20000)
    q = res.x
    qf, qb = par(q)
    a_f, a_b = _area(fun, qf), _area(fun, qb)
    cf, cb = f(q)
    return dict(bound_fraction=a_b / (a_f + a_b), free=cf, bound=cb, baseline=q[-1], free_par=list(qf), bound_par=list(qb),
                rms=float(np.sqrt(np.mean((y - cf - cb - q[-1]) ** 2))))


def hwhm(fun, q):
    """Half-width at half-maximum (cm-1) of a fitted component, numerically."""
    xx = np.arange(q[1] - 40, q[1] + 40, 0.001)
    yy = fun(xx, *q)
    above = xx[yy >= yy.max() / 2]
    return (above[-1] - above[0]) / 2
