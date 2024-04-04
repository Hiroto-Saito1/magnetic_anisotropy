#!/usr/bin/env python

"""
Description:
    Factors of discrete Fourier transform.

Usage:
    fourier.py [-h | --help]

Options:
    -h --help   Show this help screen.

Notes:
    This module is quite short but convenient to integrate the convention 
    of Fourier transform.
"""

from typing import Optional

import numpy as np

from wannier_utils.logger import get_logger

logger = get_logger(__name__)


def main():
    from itertools import product
    from pathlib import Path

    from docopt import docopt

    from wannier_utils.wannier_system import WannierSystem

    _ = docopt(__doc__)
    base_dir = Path(__file__).parents[2]
    win_file = base_dir/"tests"/"Fe.win"
    hr_dat = base_dir/"tests"/"Fe_hr.dat"
    ws = WannierSystem(hr_dat=str(hr_dat), win_file=str(win_file))

    kpts = [(np.arange(l) + s*0.5)/float(l) for l, s in zip(ws.win.mp_grid, [0, 0, 0])]
    klist = np.array(list(product(*kpts)), dtype=np.float64)
    rlist = ws.ham_r.irvec
    fourier = Fourier(klist, rlist, ndegen=ws.ham_r.ndegen)

    identity_k = np.identity(len(klist), dtype=np.complex128)
    identity_r = np.identity(len(rlist), dtype=np.complex128)
    logger.info(np.allclose(
        np.einsum("kr,rl->kl", fourier.factor, fourier.inv_factor, optimize=True), 
        identity_k, 
    ))
    logger.info(np.allclose(
        np.einsum("kr,rl->kl", fourier.factor, np.linalg.pinv(fourier.factor), optimize=True), 
        identity_k, 
    ))
    logger.info(np.allclose(
        np.einsum("rk,ks->rs", fourier.inv_factor, fourier.factor, optimize=True), 
        identity_r, 
    ))
    logger.info(np.allclose(
        np.einsum("rk,ks->rs", np.linalg.pinv(fourier.factor), fourier.factor, optimize=True), 
        identity_r, 
    ))


class Fourier:
    """
    Definition of discrete Fourier transform for a tight-binding Hamiltonian.

    Note:
        The convention of sign of the exponential is below.
            f(k) := \sum_{r}f(r)e^{2j \pi kr}                   (DFT)
            f(r) := \frac{1}{N_{k}}\sum_{k}f(k)e^{-2j \pi kr}   (inverse DFT)
    """
    def __init__(
        self, 
        k: np.ndarray, 
        r: np.ndarray, 
        ndegen: Optional[np.ndarray] = None, 
        rc: Optional[np.ndarray] = None, 
    ):
        """
        Constructor.

        Args:
            k (np.ndarray): reciprocal lattice vectors.
            r (np.ndarray): real lattice vectors.
            ndegen (Optional[np.ndarray]): degeneracies of Wigner-Seitz grid points. 
                                           defaults to None.
            rc (Optional[np.ndarray]): position of Wannier center in fractional coordinates.
                                       defaults to None.
        """
        self.k_2d = k[None, :] if k.ndim == 1 else k
        self.r_2d = r[None, :] if r.ndim == 1 else r
        self.kr = np.einsum("kd,rd->kr", self.k_2d, self.r_2d, optimize=True)
        self.ndegen = np.array([1]*len(self.r_2d), dtype=np.int64) if ndegen is None else ndegen
        if rc is None:
            self.rc_2d, self.krc = None, None; return
        self.rc_2d = rc[None, :] if rc.ndim == 1 else rc
        self.krc = np.einsum("kd,pd->kp", self.k_2d, self.rc_2d, optimize=True)

    @property
    def factor(self) -> np.ndarray:
        return self._get_factor()

    @property
    def inv_factor(self) -> np.ndarray:
        #return self._get_pinv_factor()
        return self._get_inv_factor()

    def _get_factor(self) -> np.ndarray:
        """
        Return the factor of Fourier transform (r -> k).

        Returns:
            np.ndarray: Fourier factor in (nk, nr) shape if self.rc is None, 
                        otherwise in (nk, nr, nwan, nwan) shape.
        """
        exp_kr = np.exp(2*np.pi*1j*self.kr)
        if self.krc is None:
            return np.einsum(
                "kr,r->kr", 
                exp_kr, 1/self.ndegen, 
                optimize=True, 
            )
        exp_krc = np.exp(2*np.pi*1j*self.krc)
        return exp_kr[:, :, None, None]*np.conj(exp_krc[:, None, :, None])\
               *exp_krc[:, None, None, :]

    def _get_inv_factor(self) -> np.ndarray:
        """
        Return the factor of inverse Fourier transform (k -> r).

        Returns:
            np.ndarray: inverse Fourier factor in (nr, nk) shape if self.rc is None, 
                        otherwise in (nr, nk, nwan, nwan) shape.
        """ 
        exp_mrk = np.exp(-2*np.pi*1j*self.kr.T)
        if self.krc is None:
            return exp_mrk/len(self.k_2d)
        exp_mrck = np.exp(-2*np.pi*1j*self.krc.T)
        return exp_mrck[:, None, None, :]*np.conj(exp_mrck[:, None, :, None])\
               *exp_mrk[:, :, None, None]/len(self.k_2d)

    def _get_pinv_factor(self) -> np.ndarray:
        """
        Return the factor of inverse Fourier transform by pseudo-inverse of NumPy.

        Returns:
            np.ndarray: inverse Fourier factor in pseudo-inverse shape.
        """
        factor = self._get_factor()
        if self.krc is None:
            return np.linalg.pinv(factor)
        return np.linalg.pinv(factor.transpose(2, 3, 0, 1)).transpose(2, 3, 0, 1)


if __name__ == "__main__":
    main()
