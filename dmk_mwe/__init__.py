"""Minimal working examples for the analyses of 'From Electron Density to Ionic Conductivity for the Mechanistic
Understanding of a Fast Charging Acetone Electrolyte' (Corral-Casas, Alary, Ewen, Dini).

Each module reimplements one analysis of the paper in plain NumPy/SciPy, with the same definitions and parameters as
the code used for the published numbers; the scripts in ../examples run them on the Helix dataset and compare the
output with the published values.
"""
import os

DATA = os.environ.get("DMK_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "helix_data"))


def data(*parts):
    """Path inside the Helix dataset; set DMK_DATA to where it was downloaded."""
    return os.path.join(os.path.abspath(DATA), *parts)
