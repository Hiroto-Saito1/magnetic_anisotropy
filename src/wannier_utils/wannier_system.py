#!/usr/bin/env python

from itertools import product
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from wannier_utils.hamiltonian import HamR, HamK
from wannier_utils.mag_sym import MagneticSpacegroupAnalyzer
from wannier_utils.nnkp import NNKP
from wannier_utils.win import Win
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


class WannierSystem:
    """
    contains all information from Wannier90.

    Attributes:
        use_ws_dist: 
        one_shot: whether the process was one-shot calculation.
        ham_r: Wannier-based tight-binding Hamiltonian in real space.
        nnkp: Nnkp instance.
        win: Win instance.
        num_wann: the number of wannier orbitals.
        a: real-space lattice vectors.
        b: reciprocal-space lattice vectors.
        natom: the number of atoms in a unit cell.
        volume: volume of a unit cell.
    """
    def __init__(
        self,
        tb_dat: Optional[Path] = None,
        hr_dat: Optional[Path] = None,
        nnkp_file: Optional[Path] = None,
        win_file: Optional[Path] = None,
        is_reorder: bool = False,
        use_ws_dist: bool = False,
    ):
        """
        Constructor.

        Args:
            tb_dat (Optional[Path]): Path of wannier_tb.dat.
            hr_dat (Optional[Path]): Path of wannier_hr.dat.
            nnkp_file (Optional[Path]): Path of wannier.nnkp.
            win_file (Optional[Path]): Path of wannier.win.
            is_reorder (bool): whether to reorder Hamiltonian (for VASP, Wannier90 ver.1, etc.). defaults to False.
            use_ws_dist (bool): _description_. defaults to False.
        """
        if not (tb_dat or hr_dat or nnkp_file or win_file):
            raise ValueError("No file is given.")
        if tb_dat or hr_dat:
            self.ham_r = HamR(tb_dat=tb_dat, hr_dat=hr_dat, is_reorder=is_reorder)
        if nnkp_file:
            self.nnkp = NNKP(nnkp_file)
        if win_file:
            self.win = Win(win_file)

        self.use_ws_dist = use_ws_dist
        if self.use_ws_dist:
            self._convert_ham_r()

    def calc_ham_k(
        self, 
        k: np.ndarray, 
        rc: Optional[np.ndarray] = None, 
        diagonalize: bool = False, 
    ):
        return HamK(self.ham_r, k, rc=rc, diagonalize=diagonalize)

    def set_sym(self, magmoms: Optional[Union[str, np.ndarray]] = None):
        """
        set symmetry operations of system. structure is read from win file.

        Args:
            magmoms (Optional): "x", "y", "z", or (nsites, 3) array-like.

        Attributes:
            sg_ops: symmetry operations of spacegroup.
            msg_ops: symmetry operations of magnetic spacegroup.
        """
        if not hasattr(self, "win"):
            raise ValueError("win file is not specified.")

        if magmoms is None:
            self.sym_ops = SpacegroupAnalyzer(self.win.structure).get_space_group_operations()
            return
        if magmoms == "x" or magmoms == "y" or magmoms == "z":
            magmom = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
            magmoms = [np.array(magmom[magmoms], dtype=np.float64)]*self.win.structure.num_sites
        self.win.structure.add_site_property("magmom", magmoms)
        msa = MagneticSpacegroupAnalyzer(self.win.structure)
        self.sym_ops = msa.magnetic_spacegroup_operations

    def merge(self, hrs: np.ndarray, orig_weight: float, add_weight: float):
        """
        update Hamiltonian by merging original with the given other Hamiltonian.

        Args:
            hr: a Hamiltonian in real space.
            orig_weight: superposition weight of original Hamiltonian.
            add_weight: superposition weight of the other Hamiltonian.
        """
        assert self.num_wann == hrs.shape[1], "Different numbers of Wannier orbitals."
        self.ham_r.hrs *= orig_weight
        self.ham_r += hrs*add_weight

    @property
    def num_wann(self) -> int:
        if hasattr(self, "ham_r"):
            return self.ham_r.num_wann
        if hasattr(self, "win"):
            return self.win.num_wann
        if hasattr(self, "nnkp"):
            return self.nnkp.num_wann
        raise ValueError("num_wann is not defined.")

    @property
    def a(self) -> np.ndarray:
        if hasattr(self, "nnkp"):
            return self.nnkp.a
        if hasattr(self, "ham_r") and hasattr(self.ham_r.a):
            return self.ham_r.a
        raise ValueError("a is not defined.")

    @property
    def b(self) -> np.ndarray:
        if hasattr(self, "nnkp"):
            return self.nnkp.b
        raise ValueError("b is not defined.")

    @property
    def natom(self) -> int:
        if hasattr(self, "nnkp"):
            return self.nnkp.natom
        raise ValueError("natom is not defined.")

    @property
    def volume(self) -> float:
        return np.abs(np.dot(self.a[0], np.cross(self.a[1], self.a[2])))

    def _calc_irvec(self, rd: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Args:
            rd (np.ndarray): r[1] - r[0].

        Returns:
            irvec (np.ndarray): lattice points in the Wigner-Seitz supercell.
            ir0 (int): index of origin of lattice points.
            ndegen (np.ndarray): degeneracies of the lattice points.
        """
        mp_grid = self.win.mp_grid
        mp_list = np.array(list(product(
            range(-mp_grid[0], mp_grid[0] + 1),
            range(-mp_grid[1], mp_grid[1] + 1),
            range(-mp_grid[2], mp_grid[2] + 1),
        )))
        ilist = np.array(list(product(range(-2, 3), repeat=3)))
        # all the possible vectors for transfer. size [npos, 3]
        pos = rd[None, :] + mp_list
        # get distances from Wannier orbitals on  supercell (L1*L2*L3).
        # sizes: rlist[npos, L1*L2*L3, 3], dist[npos, L1*L2*L3].
        rlist = pos[:, None, :] - np.einsum("na,a->na", ilist, mp_grid, optimize=True)[None, :, :]
        real_metric = np.einsum("id,jd->ij", self.nnkp.a, self.nnkp.a, optimize=True)
        dists = np.einsum("ija,ab,ijb->ij", rlist, real_metric, rlist, optimize=True)
        # dists[:, 62] = dists[:, i1=i2=i3=0]
        dist0 = dists[:, 62]
        inds = np.where(dist0 - np.min(dists, axis=1) < 1e-5)
        ndegen_all = np.sum(np.abs(dists - dist0[:, None]) < 1e-5, axis=1)

        irvec = mp_list[inds]
        ir0 = np.where(~irvec.any(axis=1))[0][0]
        ndegen = ndegen_all[inds]
        # check sum rule
        assert np.sum(1/ndegen) - np.product(mp_grid) < 1e-8, "Error in finding Wigner-Seitz points."
        return irvec, ir0, ndegen

    def _convert_ham_r(self):
        """
        convert HamR using distance of lattice points.
        """
        mp_grid = self.win.mp_grid
        hrs0 = np.zeros(
            (mp_grid[0], mp_grid[1], mp_grid[2], self.num_wann, self.num_wann), 
            dtype=np.complex128, 
        )
        Amnrs0 = np.zeros(
            (mp_grid[0], mp_grid[1], mp_grid[2], self.num_wann, self.num_wann, 3), 
            dtype=np.complex128, 
        )
        for ir in range(self.ham_r.nrpts):
            r = np.mod(self.ham_r.irvec[ir] + mp_grid, mp_grid)
            hrs0[r[0], r[1], r[2]] = self.ham_r.hrs[ir]
            if hasattr(self.ham_r, "Amnrs"):
                Amnrs0[r[0], r[1], r[2]] = self.ham_r.Amnrs[ir]

        nrpairs = list(product(range(self.nnkp.natom), repeat=2))
        irvec_list = []
        ndegen_list = []
        for i, (nr1, nr2) in enumerate(nrpairs):
            rd = np.array(self.nnkp.atom_pos_r[nr2]) - np.array(self.nnkp.atom_pos_r[nr1])
            irvec, _, ndegen = self._calc_irvec(rd)
            irvec_list.append(irvec)
            ndegen_list.append(ndegen)

        irvec = np.unique(np.concatenate(irvec_list), axis=0)
        nrpts = len(irvec)
        hrs = np.zeros((nrpts, self.num_wann, self.num_wann), dtype=np.complex128)
        Amnrs = np.zeros((nrpts, self.num_wann, self.num_wann, 3), dtype=np.complex128)

        nir = (2*mp_grid[1] + 1)*(2*mp_grid[2] + 1)*(irvec[:,0] + mp_grid[0]) \
               + (2*mp_grid[2] + 1)*(irvec[:,1] + mp_grid[1]) + (irvec[:,2] + mp_grid[2])
        sorted_arg = np.argsort(nir)
        irvec = irvec[sorted_arg]

        for i, (nr1, nr2) in enumerate(nrpairs):
            orb1 = [x for pos in self.nnkp.atom_pos[nr1] for x in self.nnkp.atom_orb[pos]]
            orb2 = [x for pos in self.nnkp.atom_pos[nr2] for x in self.nnkp.atom_orb[pos]]
            ir1 = 0
            for r, ndeg in zip(irvec_list[i], ndegen_list[i]):
                while not np.allclose(irvec[ir1], r):
                    ir1 += 1
                s = np.mod(r + mp_grid, mp_grid)
                for m, n in product(orb1, orb2):
                    hrs[ir1, m, n] = hrs0[s[0], s[1], s[2], m, n]/ndeg
                if hasattr(self.ham_r, "Amnrs"):
                    for m, n in product(orb1, orb2):
                        Amnrs[ir1, m, n] = Amnrs0[s[0], s[1], s[2], m, n]/ndeg
        
        self.ham_r.nrpts = nrpts
        self.ham_r.irvec = irvec
        self.ham_r.hrs = hrs
        self.ham_r.ndegen = np.ones([nrpts])
        if hasattr(self.ham_r, "Amnrs"):
            self.ham_r.Amnrs = Amnrs
 
