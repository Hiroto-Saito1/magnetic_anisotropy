#!/usr/bin/env python

from copy import deepcopy
from typing import Iterable, Optional, Union

import numpy as np
from scipy.optimize import brentq
from pymatgen.core.operations import SymmOp, MagSymmOp

from wannier_utils.temperature_spir import Temperature
from wannier_utils.wannier_kmesh import WannierKmesh
from wannier_utils.wannier_system import WannierSystem
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


class CPA(WannierKmesh):
    """
    wannierized Coherent potential approximation.
    """
    def __init__(
        self, 
        ws_list: Iterable[WannierSystem], 
        weight_list: Iterable[float], 
        onsite_pot_list: Iterable[np.ndarray], 
        temperature_K: float, 
        ne: Union[int, float], 
        kmesh: Iterable[int], 
        use_sym: bool = False, 
        magmoms: Optional[Union[str, np.ndarray]] = None, 
        wmax: float = 1000., 
        nproc: int = 1, 
    ):
        """
        Constructor.

        Args:
            ws_list (Iterable[WannierSystem]): WannierSystem instances.
            weight_list (Iterable[float]): weights of WannierSystem instances.
            onsite_pot_list (Iterable[np.ndarray]): onsite potentials.
            temperature_K (float): temperature in unit of K.
            ne (Union[int, float]): number of electrons to determine chemical potential.
            wmax (float): maximum of energy range treated in IR basis. defaults to 1000.
        """
        self.ws_list = ws_list
        self.weight_list = np.array(weight_list)
        self.onsite_pot_list = onsite_pot_list
        self.ne = ne
        self.ir0 = self.ws_list[0].ham_r.ir0
        self.num_wann = self.ws_list[0].num_wann
        self._init_cpa_hamiltonian()
        super().__init__(self.ws, kmesh, magmoms=magmoms, use_sym=use_sym, nproc=nproc)
        self.tmpr = Temperature("F", temperature_K, wmax)

        self.pot_iwn = np.zeros(
            [self.tmpr.basis.size, self.num_wann, self.num_wann], 
            dtype=np.complex128, 
        )
        self.tmat_iwn = np.zeros(
            [self.tmpr.basis.size, self.num_wann, self.num_wann], 
            dtype=np.complex128, 
        )

    def _init_cpa_hamiltonian(self):
        # shift onsite potential for each Hamiltonian
        for ws, pot in zip(self.ws_list, self.onsite_pot_list):
            assert ws.ham_r.ir0 == self.ir0, "Mismatch of ir0."
            assert ws.num_wann == self.num_wann, "Mismatch of num_wann."
            for i in range(self.num_wann):
                ws.ham_r.hrs[self.ir0, i, i] -= pot

        # set pot_list and pot_iwn
        self.pot_list = np.array([np.diag(ws.ham_r.hrs[self.ir0]) for ws in self.ws_list])
        pot = np.einsum("an,a->n", self.pot_list, self.weight_list, optimize=True)
        self.pot_iwn += np.diag(pot[:])[None, :, :]

        # set merged hopping ==> self.ws
        for i, ws in enumerate(self.ws_list):
            if i:
                self.ws.ham_r.hrs += ws.ham_r.hrs*self.weight_list[i]
            else:
                self.ws = deepcopy(ws)
                self.ws.ham_r.hrs *= self.weight_list[i]

        # we use self.pot_iwn for diagonal part, so diagonal part of ham_r should be zero
        for i in range(self.num_wann):
            self.ws.ham_r.hrs[self.ir0, i, i] = 0

    def loop(self, max_niter: int = 100, cpa_thr: float = 1e-8):
        for i in range(max_niter):
            self._calc_green_w()
            tmat_list = [self._calc_tmat(pot) for pot in self.pot_list]
            self.tmat_iwn = np.einsum(
                "inml,i->nml", 
                tmat_list, self.weight_list, 
                optimize=True, 
            )
            sum_abs = np.sum(np.abs(self.tmat_iwn))
            if sum_abs < cpa_thr:   # CPA condition
                return i
            logger.debug("iter = {:4d}, delta = {:15.8f}.".format(i, sum_abs))
            self._calc_new_pot()

    def _calc_green_kw_ksum(self):
        """
        Attributes:
            mu: determined using bisection method.
            green_w: calculated in self._calc_ne.
        """
        dmu = 30
        if not hasattr(self, "mu"): self.mu = 0
        f = lambda x: self._calc_ne(x) - self.ne
        self.mu, res = brentq(f, self.mu - dmu, self.mu + dmu, xtol=1e-10, full_output=True)
        logger.debug(
            "mu conv: {},  niter = {:3d},  mu = {:14.8f}."\
            .format(res.converged, res.iterations, self.mu)
            )
        self.green_w = self.parallel_k.apply_func_ksum(
            self._calc_green_w, 
            #np.zeros((self.tmpr.basis.size, self.num_wann, self.num_wann), dtype=np.complex128), 
        )/self.mp_points.nk

    def _calc_green_w(self, k: np.ndarray, rc: Optional[np.ndarray] = None) -> np.ndarray:
        ham_k = self.ws.calc_ham_k(k, rc=rc)
        return self.tmpr.get_green_w(ham_k.hk, self.mu, -self.pot_iwn)

    def _calc_tr_green_w(self, k: np.ndarray) -> np.ndarray:
        return np.trace(self._calc_green_w(k), axis1=1, axis2=2)

    def _recover_tr_green_w(self, result: np.ndarray, op: Union[SymmOp, MagSymmOp]) \
    -> np.ndarray:
        return result

    def _calc_ne(self, mu: float) -> float:
        self.mu = mu
        green_w = self.parallel_k.apply_func_ksum(
            self._calc_tr_green_w, 
            #np.zeros((self.tmpr.basis.size), dtype=np.complex128), 
            func_recover=self._recover_tr_green_w, 
        )/self.mp_points.nk
        green_l = self.tmpr.sample_matsu.fit(green_w)
        ne = green_l@(-self.tmpr.basis.u(self.tmpr.basis.beta))
        assert ne.imag < 1e-5, "Large imaginary part at the number of electrons."
        logger.debug(f"mu = {mu}, ne = {ne.real}.")
        return ne.real

    def _calc_tmat(self, pot: np.ndarray) -> np.ndarray:
        # tmat_a = (v_a - v_c)*(1 - self.g*(v_a - v_c))^{-1}
        v = np.diag(pot)[None, :, :] - self.pot_iwn
        tmp = np.identity(self.num_wann)[None, :, :] - np.einsum(
            "nab,nbc->nac", 
            self.green_w, v, 
            optimize=True, 
        )
        return np.einsum("nab,nbc->nac", v, np.linalg.inv(tmp), optimize=True)

    def _calc_new_pot(self):
        # self.pot += tmat*(1 + self.g * tmat)^{-1}
        tmp = np.identity(self.num_wann)[None, :, :] + np.einsum(
            "nab,nbc->nac", 
            self.green_w, self.tmat_iwn, 
            optimize=True, 
        )
        self.pot_iwn += np.einsum(
            "nab,nbc->nac", 
            self.tmat_iwn, np.linalg.inv(tmp), 
            optimize=True, 
        )
