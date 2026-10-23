"""Ionic conductivity (Onsager decomposition) and self-diffusion from mean squared displacements (Methods, 'Ionic
conductivity'; SI Section S2).

Pipeline, as used for every published conductivity:
  1. positions unwrapped, in the barycentric frame (system centre of mass removed in every frame); Li+ as atoms,
     FSI- and DMK as molecular centres of mass                                             -> `barycentric_coms`
  2. FFT mean squared displacements of each particle, and of the collective displacements M+, M-, M+ + M-, from which
     the distinct and cross terms follow                                                     -> `msd_fft`, `onsager_msds`
  3. fitting window chosen once on the charge-weighted total MSD (replica average): the longest window with R^2 >= 0.999,
     0.1-1 % of the lags long, starting in the first 20 % (self-diffusion: 20-66 % long, starting in the first 80 %,
     chosen on the cation self MSD)                                                           -> `best_window`
  4. sigma = e^2 / (6 V kB T) [N+ s++self + N- s--self + N+(N+-1) s++dist + N-(N--1) s--dist + 2 N+ N- q+ q- s+-cross]
     with every replica fitted in the same window: central value = fit of the average (= mean of the replica slopes),
     uncertainty = standard error over replicas                                              -> `conductivity`
  5. D = slope / 6 plus the Yeh-Hummer term kB T xi / (6 pi eta L)                            -> `yeh_hummer`
"""
import numpy as np
from scipy.stats import linregress

KB = 1.38064852e-23                  # J/K
E = 1.602176634e-19                  # C
XI = 2.837297                        # Yeh & Hummer (2004), cubic box
TO_MSCM = 10.0                       # S/m -> mS/cm


def msd_fft(x):
    """MSD(lag) averaged over particles, x of shape (n_frames, n_particles, 3) or (n_frames, 3); all time origins.

    MSD(m) = [sum_t |x(t+m)|^2 + |x(t)|^2 - 2 x(t+m).x(t)] / (T - m), the correlation term by zero-padded FFT."""
    x = np.asarray(x, float)
    if x.ndim == 2:
        x = x[:, None, :]
    T = x.shape[0]
    r2 = (x ** 2).sum(-1)                                            # (T, N)
    s1 = np.cumsum(r2, 0)[::-1]                                      # sum_{t <= T-1-m} |x(t)|^2
    s2 = np.cumsum(r2[::-1], 0)[::-1]                                # sum_{t >= m} |x(t)|^2
    f = np.fft.rfft(x, n=2 * T, axis=0)
    corr = np.fft.irfft(f * f.conj(), n=2 * T, axis=0)[:T].sum(-1)   # (T, N)
    sd = np.clip(s1 + s2 - 2 * corr, 0, None)
    return (sd / np.arange(T, 0, -1)[:, None]).mean(1)


def barycentric_coms(positions, masses, molecules):
    """Centres of mass of each molecule (list of atom-index arrays) after removing the system centre of mass.
    positions: unwrapped, (n_frames, n_atoms, 3)."""
    m = np.asarray(masses, float)
    x = positions - (np.einsum("a,tad->td", m, positions) / m.sum())[:, None, :]
    return np.stack([np.einsum("a,tad->td", m[ix], x[:, ix]) / m[ix].sum() for ix in molecules], axis=1)


def onsager_msds(cat, an):
    """Self (per ion), distinct (per pair) and cross (per pair) MSD terms from cation and anion trajectories
    (n_frames, N, 3), with N cations and N anions (1:1 salt)."""
    N = cat.shape[1]
    s_c, s_a = msd_fft(cat), msd_fft(an)
    Mc, Ma = cat.sum(1), an.sum(1)
    m_c, m_a, m_t = msd_fft(Mc), msd_fft(Ma), msd_fft(Mc + Ma)
    return {"self_cation": s_c, "self_anion": s_a,
            "cat_cat": (m_c - N * s_c) / (N * (N - 1)), "an_an": (m_a - N * s_a) / (N * (N - 1)),
            "cat_an": 0.5 * (m_t - m_c - m_a) / (N * N)}


def best_window(t, y, r2_min=0.999, region_end=0.20, wmin=0.001, wmax=0.01):
    """Longest window [t0, t1] with R^2 >= r2_min and a positive slope, on the grid of the published analysis
    (start step n/50, length step (max - min)/20); if none passes, the window with the best R^2."""
    n = len(t)
    lo, hi, smax = int(n * wmin), int(n * wmax), int(n * region_end)
    best, passed = None, None
    for i0 in range(0, smax, max(1, n // 50)):
        for L in range(lo, hi, max(1, (hi - lo) // 20)):
            i1 = i0 + L
            if i1 >= n:
                break
            fit = linregress(t[i0:i1], y[i0:i1])
            if fit.slope <= 0:
                continue
            r2 = fit.rvalue ** 2
            if best is None or r2 > best[0]:
                best = (r2, t[i0], t[i1 - 1])
            if r2 >= r2_min and (passed is None or L > passed[0]):
                passed = (L, t[i0], t[i1 - 1])
    return (passed[1], passed[2]) if passed else (best[1], best[2])


def slope(t, y, window):
    m = (t >= window[0]) & (t <= window[1])
    return linregress(t[m], y[m]).slope


def conductivity_terms(slopes, Nc, Na, qc, qa, V, T):
    """The five Onsager terms (mS/cm) from the MSD slopes (m^2/s) of one replica or of the average."""
    pre = E ** 2 / (6 * V * KB * T) * TO_MSCM
    terms = {"cation_self": Nc * qc ** 2 * slopes["self_cation"], "anion_self": Na * qa ** 2 * slopes["self_anion"],
             "cat_cat_distinct": Nc * (Nc - 1) * qc ** 2 * slopes["cat_cat"],
             "an_an_distinct": Na * (Na - 1) * qa ** 2 * slopes["an_an"],
             "cat_an_cross": 2 * Nc * Na * qc * qa * slopes["cat_an"]}
    terms = {k: pre * v for k, v in terms.items()}
    terms["total"] = sum(terms.values())
    return terms


def total_msd(msds, Nc, Na, qc, qa):
    return (Nc * qc ** 2 * msds["self_cation"] + Na * qa ** 2 * msds["self_anion"]
            + Nc * (Nc - 1) * qc ** 2 * msds["cat_cat"] + Na * (Na - 1) * qa ** 2 * msds["an_an"]
            + 2 * Nc * Na * qc * qa * msds["cat_an"])


def conductivity(t, replicas, Nc, Na, qc, qa, V, T):
    """replicas: list of dicts of MSD terms on the common lag grid t (s). Returns the window (s), the terms of the
    average (mean over replicas, by linearity of the fit), their standard errors over replicas, and the per-replica
    terms."""
    keys = ("self_cation", "self_anion", "cat_cat", "an_an", "cat_an")
    avg = {k: np.mean([r[k] for r in replicas], 0) for k in keys}
    w = best_window(t, total_msd(avg, Nc, Na, qc, qa))
    per = [conductivity_terms({k: slope(t, r[k], w) for k in keys}, Nc, Na, qc, qa, V, T) for r in replicas]
    mean = conductivity_terms({k: slope(t, avg[k], w) for k in keys}, Nc, Na, qc, qa, V, T)
    n = len(per)
    sem = {k: (np.std([p[k] for p in per], ddof=1) / np.sqrt(n) if n > 1 else np.nan) for k in mean}
    return w, mean, sem, per


def diffusion_window(t, msd_cation_self):
    """Self-diffusion window: as best_window, 20-66 % of the lags long, starting in the first 80 %."""
    return best_window(t, msd_cation_self, region_end=0.80, wmin=0.20, wmax=0.66)


def yeh_hummer(T, eta_Pas, L_m):
    """Finite-size correction added to every self-diffusion coefficient (m^2/s)."""
    return KB * T * XI / (6 * np.pi * eta_Pas * L_m)


def fickian_window(t, msd, alpha_tol=0.10, min_duration=100e-12, t_min_frac=0.02, t_max_frac=0.50, smooth=21,
                   fallback=(0.10, 0.60)):
    """Window used for the single MACE-FT trajectory (no replicas to average): the longest contiguous stretch with
    t in [2 %, 50 %] of the longest lag where the local slope d ln MSD / d ln t of the smoothed (21-point moving
    average) ln MSD lies within 1 +- alpha_tol, at least 100 ps long; otherwise 10-60 % of the lags.
    Returns inclusive indices (i0, i1)."""
    t, msd = np.asarray(t, float), np.asarray(msd, float)
    pos = np.where((t > 0) & (msd > 0))[0]
    lt, lm = np.log(t[pos]), np.log(msd[pos])
    if smooth > 1 and len(lm) > smooth:
        lm = np.convolve(lm, np.ones(smooth) / smooth, mode="same")
    alpha = np.gradient(lm, lt)
    ok = (np.abs(alpha - 1.0) <= alpha_tol) & (t[pos] >= t_min_frac * t[-1]) & (t[pos] <= t_max_frac * t[-1])
    best, i = None, 0
    while i < len(ok):
        if ok[i]:
            j = i + 1
            while j < len(ok) and ok[j]:
                j += 1
            if best is None or j - i > best[1] - best[0] + 1:
                best = (i, j - 1)
            i = j
        else:
            i += 1
    if best is not None and t[pos[best[1]]] - t[pos[best[0]]] >= min_duration:
        return int(pos[best[0]]), int(pos[best[1]])
    n = len(t)
    return int(n * fallback[0]), int(n * fallback[1])


def diffusion_single(t, msd, window=None):
    """D (m^2/s) from MSD = 6 D t fitted over the inclusive index window (default: fickian_window)."""
    i0, i1 = window if window is not None else fickian_window(t, msd)
    return linregress(t[i0:i1 + 1], msd[i0:i1 + 1]).slope / 6.0, (t[i0], t[i1])


def unwrap(positions, box):
    """Unwrap a cubic-box trajectory frame by frame (minimum image of each frame-to-frame step)."""
    d = np.diff(positions, axis=0)
    d -= box * np.round(d / box)
    return np.concatenate([positions[:1], positions[:1] + np.cumsum(d, 0)])
