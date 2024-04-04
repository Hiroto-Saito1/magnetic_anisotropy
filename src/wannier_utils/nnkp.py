#!/usr/bin/env python

from pathlib import Path
from traceback import format_exc

import numpy as np

from wannier_utils.logger import get_logger

logger = get_logger(__name__)


class NNKP:
    """
    information in nnkp file.

    Attributes:
        a: real-space lattice vectors.
        b: reciprocal-space lattice vectors.
        num_wann: the number of projections (Wannier orbitals) in a unit cell.
        nw2l: azimuthal quantum numbers of projections.
        nw2m: magnetic quantum numbers of projections.
        nw2r: centres of projection functions in fractional coordinates.
        atom_orb: list-in-list of indices of Wannier orbitals whose azimuthal 
                  quantum numbers agree.
        atom_pos: list-in-list of indices of atom_orb whose positions agree 
                  (e.g. atom_pos[0] = [0, 1, 2] => Wannier orbitals in 
                  atom_orb[i] exist at the same position for i = 0, 1, 2).
        atom_pos_r: list of fractional coordinates corresponding to atom_pos.
        natom: the number of atoms in a unit cell
        nw2n: atomic labels for Wannier orbitals.
    """
    def __init__(self, nnkp_file: Path):
        try:
            with open(nnkp_file, mode="r") as fp:
                lines = fp.readlines()

            self.a, self.b = np.zeros([3, 3]), np.zeros([3, 3])
            for i in range(3):
                self.a[i] = np.array([float(x) for x in lines[i + 5].split()])
                self.b[i] = np.array([float(x) for x in lines[i + 11].split()])

            for i, line in enumerate(lines):
                if ("begin projections" in line or "begin spinor_projections" in line):
                    spinors = ("begin spinor_projections" in line)
                    self.num_wann = int(lines[i + 1])
                    atom_orb_strlist = []
                    atom_pos_strlist = []
                    self.nw2l = np.zeros([self.num_wann], dtype=np.int64)
                    self.nw2m = np.zeros([self.num_wann], dtype=np.int64)
                    self.nw2r = np.zeros([self.num_wann, 3], dtype=np.float64)

                    # read projections
                    for j in range(self.num_wann):
                        if (spinors):
                            proj_str = lines[i + 2 + 3 * j]
                        else:
                            proj_str = lines[i + 2 + 2 * j]
                        proj_dat = proj_str.split()
                        self.nw2l[j] = int(proj_dat[3])
                        self.nw2m[j] = int(proj_dat[4])
                        self.nw2r[j, :] = [float(x) for x in proj_dat[0:3]]
                        atom_orb_strlist.append(proj_str[0:40])
                        atom_pos_strlist.append(proj_str[0:35])

                    # set atom_pos_r, atom_pos, atom_orb
                    # for example, in Fe case,
                    # atom_pos_r: [[0.0, 0.0, 0.0]]
                    # atom_pos: [[0, 1, 2]]
                    # atom_orb: [[0, 1], [2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
                    atom_orb_uniq = sorted(set(atom_orb_strlist), key=atom_orb_strlist.index)
                    atom_pos_uniq = sorted(set(atom_pos_strlist), key=atom_pos_strlist.index)
                    self.atom_orb = []
                    for orb_str in atom_orb_uniq:
                        indices = [j for j, x in enumerate(atom_orb_strlist) if x == orb_str]
                        self.atom_orb.append(indices)
                    self.atom_pos = []
                    self.atom_pos_r = []
                    for pos_str in atom_pos_uniq:
                        indices = [j for j, x in enumerate(atom_orb_uniq) if pos_str in x]
                        self.atom_pos.append(indices)
                        self.atom_pos_r.append([float(x) for x in pos_str.split()[0:3]])
                    self.natom = len(self.atom_pos_r)
                    self.nw2n = np.zeros([self.num_wann], dtype=np.int64)
                    for i, pos in enumerate(self.atom_pos):
                        for p in pos:
                            for n in self.atom_orb[p]:
                                self.nw2n[n] = i

        except Exception:
            logger.error(format_exc())
