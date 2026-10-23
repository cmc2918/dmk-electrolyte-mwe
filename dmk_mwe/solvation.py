"""Solvation structure: radial distribution functions, coordination numbers and anion-centric speciation (Methods,
'Solvation structure'; Table 2 and the structure columns of Tables 3-5).

Conventions (as published):
  g(r)        Li+-O(carbonyl) and Li+-O(sulfonyl), 0-10 A in 250 bins (MACE-FT, AIMD) or the stored per-replica RDFs
              (OPLS-ECC, ByteFF-Pol); normalised as MDAnalysis InterRDF(norm="rdf")             -> `rdf`
  r_peak      maximum of g in 1.4-3.0 A; r_min the minimum of g within 2 A beyond it (and below 4.5 A)
                                                                                              -> `first_peak_and_minimum`
  CN          running integral 4 pi rho int_0^r g r'^2 dr' (cumulative trapezoid) at r_min: donor O atoms, so a
              bidentate anion counts two                                                       -> `running_cn`
              OPLS-ECC/ByteFF-Pol: g averaged over replicas, r_min averaged over replicas; rho from the box edge
              used to normalise the published RDFs (examples/02, BOX). The edge can also be recovered from a contact
              list (carbonyl O within 3.0 A against the integral of g), to within 0.03 %       -> `box_edge_from_contacts`
  speciation  each FSI- classified by the number of distinct Li+ within 2.9 A of any of its sulfonyl O: 0 SSIP,
              1 CIP, >= 2 AGG; every anion of every analysed frame counts                     -> `speciation`
  O/FSI       sulfonyl O donated per bound anion: CN(sulfonyl) / anions bound per Li+
"""
import numpy as np

EDGES = np.linspace(0.0, 10.0, 251)


def mic(d, L):
    return d - L * np.round(d / L)


def rdf(frames, a_idx, b_idx, L, edges=EDGES):
    """g(r) between atom sets a and b over frames (n_frames, n_atoms, 3) in a cubic box of edge L (A)."""
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    h = np.zeros(len(r))
    for X in np.array_split(frames, max(1, len(frames) // 2000)):
        d = np.linalg.norm(mic(X[:, b_idx][:, :, None] - X[:, a_idx][:, None], L), axis=3)
        h += np.histogram(d, bins=edges)[0]
    return r, h / (len(frames) * len(a_idx) * len(b_idx) / L ** 3 * shell)


def first_peak_and_minimum(r, g, peak_lo=1.4, peak_hi=3.0, span=2.0, cap=4.5):
    m = (r >= peak_lo) & (r <= peak_hi)
    r_peak = r[m][np.argmax(g[m])]
    m2 = (r >= r_peak) & (r <= min(r_peak + span, cap))
    return float(r_peak), float(r[m2][np.argmin(g[m2])])


def running_cn(r, g, rho):
    y = 4.0 * np.pi * rho * g * r ** 2
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(r))])


def cn_at(r, g, rho, r_min):
    return float(np.interp(r_min, r, running_cn(r, g, rho)))


def box_edge_from_contacts(n_contacts_3A, n_frames, n_li, r, g, n_donor):
    """Box edge (A) for which the RDF integral up to 3.0 A reproduces the counted contacts within 3.0 A:
    rho = (contacts per Li+) / int_0^3 4 pi r^2 g dr, L = (n_donor / rho)^(1/3)."""
    per_li = n_contacts_3A / (n_frames * n_li)
    rho = per_li / float(np.interp(3.0, r, running_cn(r, g, 1.0)))
    return (n_donor / rho) ** (1.0 / 3.0)


def speciation(frame, anion, li, n_frames, n_anions):
    """SSIP/CIP/AGG (%) from contact rows (frame, anion index, Li index) already restricted to d < 2.9 A."""
    key = np.unique(np.stack([frame, anion, li], 1), axis=0)              # distinct (frame, anion, Li)
    _, n_li_per = np.unique(key[:, :2], axis=0, return_counts=True)       # Li per (frame, anion)
    h = np.bincount(np.minimum(n_li_per, 2), minlength=3).astype(float)
    h[0] = n_frames * n_anions - h[1] - h[2]
    return 100.0 * h / h.sum()


def read_xyz(path, every=1):
    """Frames of an extended-XYZ file (optionally .xz) as (n_frames, n_atoms, 3), the element list and the cubic box
    edge from the Lattice entry of the first frame."""
    import lzma
    op = lzma.open if str(path).endswith(".xz") else open
    frames, elems, L = [], None, None
    with op(path, "rt") as f:
        k = 0
        while True:
            line = f.readline()
            if not line:
                break
            n = int(line)
            head = f.readline()
            if L is None and "Lattice=" in head:
                L = float(head.split('Lattice="')[1].split()[0])
            rows = [f.readline().split() for _ in range(n)]
            if k % every == 0:
                frames.append(np.array([row[1:4] for row in rows], float))
                if elems is None:
                    elems = [row[0] for row in rows]
            k += 1
    return np.array(frames), elems, L
