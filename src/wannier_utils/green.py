#!/usr/bin/env python

"""
Description:
    Matsubara Green's function in Wannier-based tight-binding model.
"""

from typing import List

import numpy as np

from wannier_utils.temperature_spir import Temperature
from wannier_utils.wannier_system import WannierSystem
from wannier_utils.wannier_kmesh import WannierKmesh
from wannier_utils.fourier import Fourier


class Green(WannierKmesh):
    def __init__(
        self, 
        ws: WannierSystem, 
        kmesh: List[int], 
        chem_pot: float, 
        temperature_K: float, 
        nproc: int = 1, 
    ):
        super().__init__(ws, kmesh, nproc=nproc)
        self.chem_pot = chem_pot
        self.ham_ks = self.parallel_k.get_ham_ks(diagonalize=True)
        wmax = np.abs(np.array([ham_k.ek for ham_k in self.ham_ks], dtype=np.float64)).max()
        self.tmpr = Temperature("F", temperature_K, wmax)
        self.greens_kw = self._get_greens_kw()

    def get_greens_rw(self) -> np.ndarray:
        rcs = [ham_k.rc for ham_k in self.ham_ks]
        fourier = Fourier(
            self.mp_points.full_kvec, 
            self.ws.ham_r.irvec, 
            rc=rcs, 
            ndegen=self.ws.ham_r.ndegen, 
        )
        if fourier.inv_factor.ndim == 2:
            return np.einsum(
                "rk,kwmn->rwmn", 
                fourier.inv_factor, self.greens_kw, 
                optimize=True, 
            )
        if fourier.inv_factor.ndim == 4:
            return np.einsum(
                "rkmn,kwmn->rwmn", 
                fourier.inv_factor, self.greens_kw, 
                optimize=True, 
            )
        raise ValueError("Factor of inverse Fourier transform has wrong shape.")

    def _get_greens_kw(self) -> np.ndarray:
        """
        Return Green's functions in reciprocal space.

        Returns:
            np.ndarray: Green's functions in (nk, nw, nwan, nwan) shape.
        """
        identity = np.identity(self.ws.num_wann, dtype=np.complex128)
        hks_mu = np.array([ham_k.hk - self.chem_pot*identity for ham_k in self.ham_ks])
        inv_greens_kw = 1j*self.tmpr.omega_sp[None, :, None, None]*identity[None, None, :, :] \
                        - hks_mu[:, None, :, :]
        return np.linalg.inv(inv_greens_kw)
