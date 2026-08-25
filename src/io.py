"""Structure and energy loading for the v-SiO2 ensembles."""
import glob
import os
import re

import numpy as np
from pymatgen.core import Structure

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
POSCAR_DIR = os.path.join(DATA_DIR, 'all_sizes')

SIZES = (24, 36, 48)
_ENERGY_RE = re.compile(r'Energy\s*=\s*(-?\d+\.?\d*)')


def structure_ids(size=None, folder=POSCAR_DIR):
    """All structure ids present on disk, e.g. '24_597'. Sorted numerically."""
    pattern = f'{size}_*.vasp' if size else '*.vasp'
    ids = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(folder, pattern))]
    return sorted(ids, key=lambda s: (int(s.split('_')[0]), int(s.split('_')[1])))


def size_of(struct_id):
    return int(struct_id.split('_')[0])


def load_structure(struct_id, folder=POSCAR_DIR):
    """Parse one .vasp file. Raises on malformed input rather than returning None."""
    return Structure.from_file(os.path.join(folder, f'{struct_id}.vasp'))


def load_energy(struct_id, folder=POSCAR_DIR):
    """Total energy in Ry, read from the POSCAR comment line."""
    path = os.path.join(folder, f'{struct_id}.vasp')
    with open(path) as fh:
        header = fh.readline()
    match = _ENERGY_RE.search(header)
    if not match:
        raise ValueError(f'no energy in header of {path}: {header!r}')
    return float(match.group(1))


def load_energies(struct_ids, folder=POSCAR_DIR, per_atom=True):
    """Energies in Ry, per atom by default (the convention used for Boltzmann weights)."""
    out = {}
    for sid in struct_ids:
        energy = load_energy(sid, folder)
        out[sid] = energy / size_of(sid) if per_atom else energy
    return out


def validate(folder=POSCAR_DIR):
    """Parse every file, returning (ok_ids, {bad_id: reason}).

    Used as a preflight so a malformed file fails loudly here instead of being
    silently skipped inside a populate loop.
    """
    ok, bad = [], {}
    for sid in structure_ids(folder=folder):
        try:
            st = load_structure(sid, folder)
            n_si = sum(1 for s in st if s.specie.symbol == 'Si')
            n_o = sum(1 for s in st if s.specie.symbol == 'O')
            if n_o != 2 * n_si:
                bad[sid] = f'non-stoichiometric Si{n_si}O{n_o}'
            elif st.volume <= 1.0:
                bad[sid] = f'degenerate cell, V={st.volume:.3f}'
            else:
                load_energy(sid, folder)
                ok.append(sid)
        except Exception as exc:
            bad[sid] = f'{type(exc).__name__}: {exc}'
    return ok, bad
