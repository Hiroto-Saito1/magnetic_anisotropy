#!/usr/bin/env python

from itertools import product
from pathlib import Path
from traceback import format_exc
from typing import Any, Callable, List, Optional

import numpy as np
from scipy.constants import physical_constants
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.sites import PeriodicSite
from pymatgen.core.structure import Structure
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


a_bohr = physical_constants["Bohr radius"][0] * 10**10


def main():
    base_dir = Path(__file__).parents[2]
    win_file = base_dir/"tests"/"Fe.win"
    w = Win(win_file)
    logger.debug(w.nw2elem)
    logger.debug(w.nw2mass)
    logger.debug(w.nw2atom_num)


class Win:
    """
    Information in win file.

    Attributes:
        num_bands: the number of bands passed to Wannier90.
        num_wann: the number of Wannier orbitals.
        num_iter: the number of iteration to the maximum localization.
        mp_grid: Monkhorst-Pack grids of k-points.
        proj_list: list of projection functions for elements.
        atom_list: list of atoms.
        atom_pos_list: list of atomic positions in fractional coordinates.
        spinors: whether Wannier orbitals are spinors
        a: real-space lattice vectors.
        nw2elem: list of atomic species of projections in pymatgen Element.
        nw2orb: list of azimuthal quantum numbers of projections.
        nw2atom_num: list of indices of atom_list for projections 
                     (e.g. nw2atom_num[i] = j => i-th projection corresponds to 
                     atom of atom_list[j]).
        nw2mass: list of atomic mass for projections.
        structure: pymatgen Structure from win file.
    """
    def __init__(self, win_file: Path):
        assert win_file.is_file(), "win file is not found."
        with open(win_file, mode="r") as fp:
            lines = fp.readlines()

        self.num_bands = self._get_param_keyword(lines, "num_bands", 0, dtype=int, is_split=True)
        self.num_wann = self._get_param_keyword(lines, "num_wann", 0, dtype=int, is_split=True)
        self.num_iter = self._get_param_keyword(lines, "num_iter", 100, dtype=int, is_split=True)

        p = self._get_param_keyword(lines, "mp_grid", dtype=str)
        assert p is not None, "mp_grid is not defined."
        self.mp_grid = np.array([int(x) for x in p.split()])

        self.proj_list = []
        self.atom_list = [] # order of atom_list is not the same as the orbital list
        self.atom_pos_list = []
        self.spinors = False
        try:
            for i, line in enumerate(lines):
                if ("begin projections" in line):
                    j = 0
                    while (not ("end projections" in lines[i + j + 1])):
                        self.proj_list.append(lines[i + j + 1])
                        j += 1

                if ("begin unit_cell_cart" in line):
                    if ("ang" in lines[i + 1]):
                        nshift, factor = 1, 1
                    elif ("bohr" in lines[i + 1]):
                        nshift, factor = 1, a_bohr
                    else:
                        nshift, factor = 0, 1
                    self.a = np.zeros([3, 3])
                    for j in range(3):
                        self.a[j] = np.array([float(x) for x in lines[i + j + 1 + nshift].split()])
                    self.a *= factor

                if ("begin atoms_frac" in line):
                    j = 0
                    while (not ("end atoms_frac" in lines[i + j + 1])):
                        self.atom_list.append((lines[i + j + 1])[0:3].strip())
                        self.atom_pos_list.append([float(x) for x in (lines[i + j + 1]).split()[1:4]])
                        j += 1

                if ("spinors" in line):
                    flag = line.split("=")[1]
                    self.spinors = ("t" in flag.lower())

        except Exception:
            logger.error(format_exc())

        self._analyze_proj()

    def _get_param_keyword(
        self, 
        lines: List[str], 
        keyword: str, 
        default_value: Optional[Any] = None, 
        dtype: Callable = int, 
        is_split: bool = False, 
    ):
        data = None
        for line in lines:
            if line.startswith(keyword):
                assert data is None, f"{keyword} is defined more than once."
                if len(line.split("=")) > 1:
                    data = line.split("=")[1]
                elif len(line.split(":")) > 1:
                    data = line.split(":")[1]
                if is_split:
                    data = data.split()[0]
        if data is None:
            data = default_value
        if data is None:
            return None
        return dtype(data)

    def _analyze_proj(self):
        self.nw2elem = []
        self.nw2orb = []
        self.nw2atom_num = []
        for proj, (i, atom) in product(self.proj_list, enumerate(self.atom_list)):
            elem = Element(atom)
            ar = proj.split(":")
            if (atom == ar[0]):
                for orb in ar[1].split(";"):
                    if ("=" in orb):
                        if ("," in orb):
                            raise NotImplementedError
                        l = int(orb.split("=")[1])
                        self._set_proj(elem, i, l)
                    else:
                        for orb_str in orb.split(","):
                            if (orb_str.strip() == "s"):
                                self._set_proj(elem, i, 0)
                            if (orb_str.strip() == "p"):
                                self._set_proj(elem, i, 1)
                            if (orb_str.strip() == "d"):
                                self._set_proj(elem, i, 2)

        self.nw2mass = [elem.atomic_mass for elem in self.nw2elem]

    def _set_proj(self, elem: Element, i: int, l: int):
        ndeg = 2*l + 1
        if (self.spinors):
            ndeg *= 2
        self.nw2elem += [elem]*ndeg
        self.nw2orb += [l]*ndeg
        self.nw2atom_num += [i]*ndeg

    @property
    def structure(self):
        """
        Return pymatgen.core.structure.Structure from the win file.
        """
        lattice = Lattice(self.a)
        sites = [PeriodicSite(sp, pos, lattice) for sp, pos in zip(self.atom_list, self.atom_pos_list)]
        return Structure.from_sites(sites)


if __name__ == "__main__":
    main()
