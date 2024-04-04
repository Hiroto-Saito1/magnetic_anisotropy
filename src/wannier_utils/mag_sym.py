#!/usr/bin/env python

from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp
from pymatgen.core.structure import IStructure, Structure
from pymatgen.core.sites import PeriodicSite
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, SpacegroupOperations

from wannier_utils.win import Win
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


def main():
    base_dir = Path(__file__).parents[2]
    win_file = base_dir/"tests"/"Fe.win"
    win = Win(win_file)
    st = win.structure
    magmoms = [Magmom(np.array([0, 0, 1])) for _ in range(st.num_sites)]
    st.add_site_property("magmom", magmoms)
    msa = MagneticSpacegroupAnalyzer(st)
    _ = [logger.debug(op) for op in msa.spacegroup_operations]
    _ = [logger.debug(mop) for mop in msa.magnetic_spacegroup_operations]


class MagneticSpacegroupAnalyzer:
    """
    determines magnetic spacegroup and its symmetry operations.
    """
    def __init__(
        self,
        st_mag: Union[IStructure, Structure],
        symprec: float = 1e-2,
        angle_tol: float = 5.0,
        magmom_tol: float = 10.0,
    ):
        """
        Constructor.

        Args:
            st_mag (Union[IStructure, Structure]): structure to determine magnetic spacegroup.
            symprec (float): site tolerance for symmetry finding by SpacegroupAnalyzer. 
                             defaults to 1e-2.
            angle_tol (float): angle tolerance for symmetry finding by SpacegroupAnalyzer. 
                               defaults to 5.0 in unit of degree.
            magmom_tol (float): angle tolerance for the operated magnetic moments. 
                                defaults to 10.0 in unit of degree.
        """
        self.st_mag = st_mag
        self.symprec = symprec
        self.angle_tol = angle_tol
        self.magmom_rad_tol = np.deg2rad(magmom_tol)

        self._sg_ops = self._get_spacegroup_operations()
        self._msp_ops = self._get_magnetic_spacegroup_operations()

    @property
    def spacegroup_operations(self) -> SpacegroupOperations:
        return self._sg_ops

    @property
    def magnetic_spacegroup_operations(self) -> List[MagSymmOp]:
        return self._msp_ops

    def _get_spacegroup_operations(self) \
    -> SpacegroupOperations:
        st_nonmag = self.st_mag.copy(
            site_properties={"magmom": [Magmom(0)]*self.st_mag.num_sites},
        )
        return SpacegroupAnalyzer(
            st_nonmag,
            symprec=self.symprec,
            angle_tolerance=self.angle_tol,
        ).get_space_group_operations()

    def _get_magnetic_spacegroup_operations(self) -> List[MagSymmOp]:
        msg_ops = []
        symmops_paramag = [MagSymmOp.from_symmop(op, t) 
                           for op, t in product(self._sg_ops, [1, -1])]
        for mop in symmops_paramag:
            is_msg_op = True
            for s in self.st_mag.sites:
                test_site = self._apply_mag_op_site(mop, s)
                if not self._check_site(test_site, self.st_mag.sites):
                    is_msg_op = False
                    break
            if is_msg_op:
                msg_ops.append(mop)
        return msg_ops

    def _apply_mag_op_site(
        self,
        mag_op: MagSymmOp,
        periodic_site: PeriodicSite,
    ) -> PeriodicSite:
        lattice = periodic_site.lattice
        new_frac_coords = mag_op.operate(periodic_site.frac_coords)
        new_magmom = self.operate_magmom_frac(
            periodic_site.properties["magmom"],
            lattice,
            mag_op,
        )
        return PeriodicSite(
            periodic_site.species,
            new_frac_coords,
            lattice,
            properties={"magmom": new_magmom},
        )

    @staticmethod
    def operate_magmom_frac(magmom: Magmom, lattice: Lattice, mag_op: MagSymmOp) \
    -> Magmom:
        magmom_crst = magmom.get_moment_relative_to_crystal_axes(lattice)
        magmom_crst_rot = mag_op.apply_rotation_only(magmom_crst) \
                          *np.linalg.det(mag_op.rotation_matrix) \
                          *mag_op.time_reversal
        return Magmom.from_moment_relative_to_crystal_axes(
            magmom_crst_rot,
            lattice,
        )

    def _check_site(
        self,
        test_site: PeriodicSite,
        other_sites: Iterable[PeriodicSite],
    ) -> bool:
        test_magmom: Magmom = test_site.properties["magmom"]
        test_magmom_cart = test_magmom.moment
        test_magmom_norm = np.linalg.norm(test_magmom_cart, ord=2)

        for s in other_sites:
            if test_site.is_periodic_image(s, tolerance=self.symprec):
                magmom: Magmom = s.properties["magmom"]
                magmom_cart = magmom.moment
                magmom_norm = np.linalg.norm(magmom_cart, ord=2)

                is_colinear = np.linalg.norm(np.cross(test_magmom_cart, magmom_cart)) \
                    <= test_magmom_norm*magmom_norm*np.sin(self.magmom_rad_tol)
                if is_colinear:
                    return np.dot(test_magmom_cart, magmom_cart) >= 0
                return False
        raise ValueError(
            "Invalid symmetry operation for site ",
            "[{:.6f}, {:.6f}, {:.6f}].".format(*test_site._frac_coords),
        )

    def get_ahe_active_dir(
        self,
        cand_dir_frac: Dict[str, List[float]] = {
            "a": [1.0, 0.0, 0.0], 
            "b": [0.0, 1.0, 0.0], 
            "c": [0.0, 0.0, 1.0], 
            "a+b": [1.0, 1.0, 0.0], 
            "a-b": [1.0, -1.0, 0.0], 
            "b+c": [0.0, 1.0, 1.0], 
            "b-c": [0.0, 1.0, -1.0], 
            "c+a": [1.0, 0.0, 1.0], 
            "c-a": [-1.0, 0.0, 1.0], 
        },
        check_break_ops: bool = False,
    ) -> Dict[str, MagSymmOp]:
        lattice = self.st_mag.lattice
        rtol_in_prod = np.cos(self.magmom_rad_tol)

        ahe_active_dirs = []
        break_ops_tot = {}
        for k, d in cand_dir_frac.items():
            m = Magmom.from_moment_relative_to_crystal_axes(d, lattice)
            m_norm = np.linalg.norm(m.moment, ord=2)
            break_ops = {}
            for i, mop in enumerate(self._msp_ops):
                m_opd = self.operate_magmom_frac(m, lattice, mop)
                if np.dot(m.moment, m_opd.moment) < m_norm*rtol_in_prod:
                    break_ops["{}_{:d}".format(k, i)] = mop
            if not break_ops.values():
                ahe_active_dirs.append(k)
            break_ops_tot.update(break_ops)

        if check_break_ops:
            logger.debug(break_ops_tot)
        return ahe_active_dirs

    def get_msg_ops_for_k(
        self,
        kpt: np.ndarray,
        is_crst: bool = True,
        rtol_in_prod: float = 1e-2,
        check_break_ops: bool = False,
    ) -> List[str]:
        lattice = self.st_mag.lattice
        if not is_crst:
            kpt = lattice.get_fractional_coords(kpt)
        kpts_app = [mop.apply_rotation_only(kpt) for mop in self._msp_ops]

        kpt_cart = lattice.get_cartesian_coords(kpt)
        kpts_app_cart = [lattice.get_cartesian_coords(k) for k in kpts_app]
        kpt_norm = np.linalg.norm(kpt_cart, ord=2)
        kpts_inpro = [np.dot(kpt_cart, k) for k in kpts_app_cart]
        is_matches = np.isclose(kpts_inpro, kpt_norm, rtol=rtol_in_prod)

        preserve_ops = []
        break_ops = []
        for is_match, mop in zip(is_matches, self._msp_ops):
            if is_match:
                preserve_ops.append(mop)
                continue
            break_ops.append(mop)

        if check_break_ops:
            logger.debug(break_ops)
        return preserve_ops


if __name__ == "__main__":
    main()
