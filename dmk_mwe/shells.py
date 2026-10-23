"""Dynamics of the Li+ shell: residence times, Transport Index and ligand-exchange events (Methods, 'Residence times and
transport mechanisms'; SI Sections S6, S7; Fig. 5).

Input: a contact table, i.e. every Li+-ligand pair closer than 4.2 A in every frame -- rows (frame, li, mol, d), with d
the Li-O(carbonyl) distance for DMK and the shortest Li-O(sulfonyl) distance for FSI- -- plus the Li+ positions and
the Li-O RDFs (the .npz files of the deposit, or `contact_table` from coordinates).

  residence time   bound indicator H(t) = 1 if d < r_min (first minimum of the model's RDF); its autocorrelation
                   C(t) = <H(t)H(0)> / <H(0)^2> (SolvationAnalysis Residence) fitted to a exp(-t/tau) + c over all lags,
                   p0 = (1, 0.1, 0.01) in frame units                                  -> `acf`, `residence_time`
  shells (TI, S7)  two-cutoff hysteresis: bind when d < r_in (half-maximum of the first RDF peak on its outer side),
                   unbind when d > r_out (3.4 A DMK, 3.2 A FSI-; OPLS-ECC: each replica's own first minimum,
                   3.62-3.70 / 3.34-3.38 A) or when the pair leaves the table                       -> `hysteresis_bound`
  Transport Index  per Li+: sum of squared 5 ps steps in which its (n_DMK, n_FSI) changes / sum of all squared steps;
                   mean over ions (and over replicas)                                    -> `transport_index`
  exchange events  coordination n(t) of every Li+ under the hysteresis shells; excursions above / below its most
                   probable value n0 classified as completed exchanges (associative / dissociative) or failed
                   attempts; same-frame swaps; tau_ex = CN / departures in completed exchanges       -> `exchanges`
"""
from collections import Counter
import numpy as np
from scipy.optimize import curve_fit

SP = ("DMK", "FSI")
R_OUT_TI = {"DMK": 3.4, "FSI": 3.2}
R_TRACK, EDGES = 4.2, np.arange(0.0, 6.0 + 0.02, 0.04)


# ------------------------------------------------------------------ cutoffs
def first_peak(r, g, lo=1.5, hi=5.5):
    m = (r >= lo) & (r <= hi)
    i = np.argmax(g[m])
    return float(r[m][i]), float(g[m][i])


def first_minimum(r, g, r_peak, lo=0.2, hi=2.0):
    m = (r > r_peak + lo) & (r < r_peak + hi)
    return float(r[m][np.argmin(g[m])])


def r_in_halfmax(r, g, r_peak, g_peak, r_out):
    m = (r > r_peak) & (r < r_out)
    rm, gm, half = r[m], g[m], g_peak / 2.0
    for i in range(len(gm) - 1):
        if gm[i] >= half and gm[i + 1] < half:
            return float(rm[i] + (gm[i] - half) / (gm[i] - gm[i + 1]) * (rm[i + 1] - rm[i]))
    return float((r_peak + r_out) / 2.0)


def cutoffs(z, variant="ti"):
    """{species: {r_in, r_out}} from the RDFs stored with a contact table; variant 'ti': r_out = 3.4 / 3.2 A,
    'own_min': r_out = first minimum of the model's RDF."""
    r, out = z["rdf_r"], {}
    for sp in SP:
        g = z[f"rdf_{sp.lower()}"]
        rp, gp = first_peak(r, g)
        r_out = R_OUT_TI[sp] if variant == "ti" else first_minimum(r, g, rp)
        out[sp] = dict(r_in=r_in_halfmax(r, g, rp, gp, r_out), r_out=float(r_out))
    return out


# ------------------------------------------------------------------ shells
def rows(z, sp, stride=1):
    f, li, mol, d = (z[f"{sp.lower()}_{k}"].astype(np.int64 if k != "d" else np.float64) for k in ("frame", "li", "mol", "d"))
    if stride > 1:
        m = f % stride == 0
        f, li, mol, d = f[m] // stride, li[m], mol[m], d[m]
    return f, li, mol, d


def hysteresis_bound(rw, r_in, r_out, n_mol):
    """Bound rows (key, frame, li, mol), sorted by pair and frame."""
    frame, li, mol, d = rw
    key = li * n_mol + mol
    o = np.lexsort((frame, key))
    key, frame, li, mol, d = key[o], frame[o], li[o], mol[o], d[o]
    if len(key) == 0:
        return key, frame, li, mol
    start = np.ones(len(key), bool)
    start[1:] = (key[1:] != key[:-1]) | (frame[1:] - frame[:-1] > 1)
    e = np.where(d < r_in, 1, np.where(d > r_out, -1, 0)).astype(np.int8)
    e[start] = np.where(d[start] < r_in, 1, -1)
    last = np.maximum.accumulate(np.where(e != 0, np.arange(len(e)), -1))
    b = e[last] == 1
    return key[b], frame[b], li[b], mol[b]


def single_cutoff_bound(rw, r_c, n_mol):
    frame, li, mol, d = rw
    m = d < r_c
    key = li[m] * n_mol + mol[m]
    o = np.lexsort((frame[m], key))
    return key[o], frame[m][o], li[m][o], mol[m][o]


def episodes(key, frame, li):
    """Contiguous bound runs per pair -> li, first frame, last frame."""
    if len(key) == 0:
        z = np.zeros(0, np.int64)
        return z, z, z
    brk = np.ones(len(key), bool)
    brk[1:] = (key[1:] != key[:-1]) | (frame[1:] - frame[:-1] > 1)
    s = np.nonzero(brk)[0]
    e = np.append(s[1:], len(key)) - 1
    return li[s], frame[s], frame[e]


# ------------------------------------------------------------------ Transport Index
def transport_index(z, stride, cut=None):
    """Mean over Li+ of the structural share of the squared displacement, steps of stride x the table's frame."""
    cut = cut or cutoffs(z, "ti")
    n_li = int(z["n_li"])
    T = (int(z["n_frames"]) - 1) // stride + 1
    S = np.zeros((T - 1, n_li), bool)
    for sp in SP:
        _, fr, li, _ = hysteresis_bound(rows(z, sp, stride), cut[sp]["r_in"], cut[sp]["r_out"], int(z[f"n_{sp.lower()}"]))
        cn = np.zeros((T, n_li), np.int32)
        np.add.at(cn, (fr, li), 1)
        S |= np.diff(cn, axis=0) != 0
    x = z["li_pos"][::stride].astype(np.float64)
    d = np.diff(x, axis=0)
    L = np.asarray(z["box"][:3], float)
    d -= L * np.round(d / L)
    dr2 = (d ** 2).sum(-1)
    return float(np.mean((dr2 * S).sum(0) / dr2.sum(0)))


# ------------------------------------------------------------------ residence times
def acf(key, frame, n_frames, batch=256):
    """<H(t)H(t+tau)> / <H(t)H(t)> summed over the pairs that are ever bound, all lags (= SolvationAnalysis)."""
    T, nfft = n_frames, 1
    while nfft < 2 * T:
        nfft *= 2
    A = np.zeros(T)
    uk, inv = np.unique(key, return_inverse=True)
    for b0 in range(0, len(uk), batch):
        m = (inv >= b0) & (inv < b0 + batch)
        H = np.zeros((min(batch, len(uk) - b0), T), np.float32)
        H[inv[m] - b0, frame[m]] = 1.0
        F = np.fft.rfft(H, n=nfft, axis=1)
        A += np.fft.irfft(F * np.conj(F), n=nfft, axis=1)[:, :T].sum(0)
    A /= (T - np.arange(T))
    return A / A[0]


def residence_time(C, dt):
    """tau (ps) of the SolvationAnalysis fit a exp(-b k) + c, p0 = (1, 0.1, 0.01), over all lags k."""
    k = np.arange(len(C))
    p, _ = curve_fit(lambda t, a, b, c: a * np.exp(-b * t) + c, k, C, p0=(1, 0.1, 0.01), maxfev=20000)
    return dt / p[1]


# ------------------------------------------------------------------ exchange events
def _runs(mask):
    T, n = mask.shape
    pad = np.zeros((1, n), np.int8)
    d = np.diff(np.vstack([pad, mask.astype(np.int8), pad]), axis=0)
    s_t, s_l = np.nonzero(d == 1)
    e_t, e_l = np.nonzero(d == -1)
    o1, o2 = np.lexsort((s_t, s_l)), np.lexsort((e_t, e_l))
    return s_l[o1], s_t[o1], e_t[o2] - 1


class _Index:
    def __init__(self, li, fr, sp, mol, T):
        k = li.astype(np.int64) * T + fr
        o = np.argsort(k, kind="stable")
        self.k, self.sp, self.mol, self.T = k[o], sp[o], mol[o], T

    def between(self, li, f0, f1):
        a = np.searchsorted(self.k, li * self.T + f0, "left")
        b = np.searchsorted(self.k, li * self.T + f1, "right")
        return {(int(self.sp[i]), int(self.mol[i])) for i in range(a, b)}

    def at(self, li, fr):
        return self.between(li, fr, fr)


def exchanges(z, stride, cut=None):
    """Exchange statistics of one contact table under the hysteresis shells (SI Table S7)."""
    cut = cut or cutoffs(z, "ti")
    n_li, T0 = int(z["n_li"]), int(z["n_frames"])
    T, dt = (T0 - 1) // stride + 1, float(z["dt"]) * stride
    n = np.zeros((T, n_li), np.int16)
    ep = []
    for si, sp in enumerate(SP):
        key, fr, li, mol = hysteresis_bound(rows(z, sp, stride), cut[sp]["r_in"], cut[sp]["r_out"], int(z[f"n_{sp.lower()}"]))
        np.add.at(n, (fr, li), 1)
        if len(key):
            brk = np.ones(len(key), bool)
            brk[1:] = (key[1:] != key[:-1]) | (fr[1:] - fr[:-1] > 1)
            s = np.nonzero(brk)[0]
            e = np.append(s[1:], len(key)) - 1
            ep.append(np.column_stack([np.full(len(s), si), li[s], mol[s], fr[s], fr[e]]))
    sp_, li_, mol_, st_, en_ = np.vstack(ep).astype(np.int64).T
    starts = _Index(li_[st_ > 0], st_[st_ > 0], sp_[st_ > 0], mol_[st_ > 0], T)
    ends = _Index(li_[en_ < T - 1], en_[en_ < T - 1], sp_[en_ < T - 1], mol_[en_ < T - 1], T)
    pn = np.bincount(n.ravel(), minlength=8)[:8] / n.size
    n0 = int(np.argmax(pn))
    li_ns = n_li * T * dt / 1000.0
    cn = {SP[s]: float((en_[sp_ == s] - st_[sp_ == s] + 1).sum() / (T * n_li)) for s in (0, 1)}
    over, under, swaps, dep = Counter(), Counter(), Counter(), Counter()
    for kind, mask, res in (("over", n > n0, over), ("under", n < n0, under)):
        Ls, t0, t1 = _runs(mask)
        ok = (t0 > 0) & (t1 < T - 1)
        for li, a, b in zip(Ls[ok], t0[ok], t1[ok]):
            li, a, b = int(li), int(a), int(b)
            if n[a - 1, li] != n0 or n[b + 1, li] != n0:
                continue
            res["formed"] += 1
            if kind == "over":
                came, left = starts.between(li, a, b), ends.at(li, b)
                old = left - came
                if old:
                    res["success"] += 1
                    for o in old:
                        for c in came:
                            res[f"{SP[o[0]]} out <- {SP[c[0]]} in"] += 1
                else:
                    res["failed"] += 1
            else:
                left, came = ends.between(li, a - 1, b), starts.at(li, b + 1)
                new = came - left
                if new:
                    res["success"] += 1
                    for o in left:
                        for c in new:
                            res[f"{SP[o[0]]} out -> {SP[c[0]]} in"] += 1
                else:
                    res["failed"] += 1
    m = en_ < T - 1
    for s, li, e in zip(sp_[m], li_[m], en_[m]):
        if n[e, li] == n0 and n[e + 1, li] == n0:
            for c in starts.at(int(li), int(e) + 1):
                swaps[f"{SP[s]} out / {SP[c[0]]} in"] += 1
    n_ex = over["success"] + under["success"] + sum(swaps.values())
    for blk in (over, under, swaps):                    # departures in completed exchanges, per route
        for k_, v in blk.items():
            if " out" in k_:
                dep[k_.split(" out")[0]] += v
    return dict(dt_ps=dt, n0=n0, P_n=pn, li_ns=li_ns, cn=cn, over=dict(over), under=dict(under), swaps=dict(swaps),
                five_time=float(pn[n0 + 1:].sum()), formed_per_ns=over["formed"] / li_ns,
                success=over["success"] / max(1, over["success"] + over["failed"]), exchanges_per_ns=n_ex / li_ns,
                routes=(over["success"] / n_ex, sum(swaps.values()) / n_ex, under["success"] / n_ex),
                tau_ex_ps={s: 1000 * cn[s] / (dep[s] / li_ns) for s in SP})


def dG_five(P_n, n0=4, T=298.15):
    """Free energy (kJ/mol) of a five- relative to a four-ligand shell, -kB T ln(P5 / P4)."""
    return float(-0.0083145 * T * np.log(P_n[n0 + 1] / P_n[n0]))


def entries_exits(z, sp, stride, r_c):
    """Entries + exits of species sp per Li+ per 100 ps (single cutoff r_c), the count compared with AIMD, and the
    number of molecules bound per Li+ at the same cutoff."""
    key, fr, li, _ = single_cutoff_bound(rows(z, sp, stride), r_c, int(z[f"n_{sp.lower()}"]))
    T = (int(z["n_frames"]) - 1) // stride + 1
    _, s, e = episodes(key, fr, li)
    k = int((s > 0).sum() + (e < T - 1).sum())
    span = (T - 1) * float(z["dt"]) * stride
    return k / int(z["n_li"]) / (span / 100.0), len(key) / (T * int(z["n_li"])), k


def contact_table(X, L, li, carb_o, sulf_o_by_anion, dt):
    """Contact table (the format of the deposited .npz) from coordinates X (n_frames, n_atoms, 3) in a cubic box L (A)."""
    T = len(X)
    mic = lambda d: d - L * np.round(d / L)
    d_dmk = np.linalg.norm(mic(X[:, carb_o][:, None] - X[:, li][:, :, None]), axis=-1)             # T, Li, DMK
    so = np.concatenate(sulf_o_by_anion)
    d_so = np.linalg.norm(mic(X[:, so][:, None] - X[:, li][:, :, None]), axis=-1)
    d_fsi = d_so.reshape(T, len(li), len(sulf_o_by_anion), -1).min(-1)
    z = {"n_li": len(li), "n_dmk": len(carb_o), "n_fsi": len(sulf_o_by_anion), "n_frames": T, "dt": dt,
         "li_pos": X[:, li], "box": np.array([L, L, L, 90, 90, 90])}
    for sp, d in (("dmk", d_dmk), ("fsi", d_fsi)):
        f, l, m = np.nonzero(d < R_TRACK)
        z.update({f"{sp}_frame": f, f"{sp}_li": l, f"{sp}_mol": m, f"{sp}_d": d[f, l, m]})
    r = 0.5 * (EDGES[1:] + EDGES[:-1])
    shell = 4.0 / 3.0 * np.pi * (EDGES[1:] ** 3 - EDGES[:-1] ** 3)
    z["rdf_r"] = r
    z["rdf_dmk"] = np.histogram(d_dmk, bins=EDGES)[0] / (T * len(li) * len(carb_o) / L ** 3 * shell)
    z["rdf_fsi"] = np.histogram(d_so, bins=EDGES)[0] / (T * len(li) * len(so) / L ** 3 * shell)
    return z
