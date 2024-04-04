#!/usr/bin/env python

from io import TextIOWrapper
from typing import Iterable, Optional, Tuple, Union

import numpy as np
from scipy.constants import elementary_charge, hbar

from wannier_utils.logger import get_logger
from wannier_utils.wannier_kmesh import WannierKmesh
from wannier_utils.wannier_system import WannierSystem

logger = get_logger(__name__)


def main():
    from pathlib import Path
    from time import perf_counter

    base_dir = Path(__file__).parents[2]
    ws = WannierSystem(
        hr_dat=base_dir/"tests"/"Fe_hr.dat", 
        nnkp_file=base_dir/"tests"/"Fe.nnkp",
    )

    start_time = perf_counter()
    kmesh = [10, 10, 10]
    ef = 17.4058
    mu_range = ef + np.arange(-5.0, 5.0001, 0.01)
    tmpr_range = np.linspace(100, 500, 5, dtype=np.float64)
    bz = Boltzmann(ws, kmesh, nproc=1)
    bz.calc_all(mu_range, tmpr_range, tau=1)
    bz.save("Fe")

    end_time = perf_counter()
    logger.info(f"elapsed_time: {end_time - start_time} [sec]")


class Boltzmann(WannierKmesh):
    """
    Boltzmann equation solution.
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        kmesh: Iterable[int], 
        magmoms: Optional[Union[str, np.ndarray]] = None, 
        use_sym: bool = False, 
        nproc: int = 1, 
    ):
        super().__init__(ws, kmesh, use_sym=use_sym, magmoms=magmoms, nproc=nproc)

    def calc_all(self, mu_range: np.ndarray, tmpr_range: np.ndarray, tau: float = 10.)\
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            mu_range (np.nparray): chemical potentials in unit of eV.
            tmpr_range (np.nparray): temperatures in unit of K.
            tau (float): relaxation time in unit of fs. default is 10.

        Returns:
            sigma: (e**2/V) sum_n v_ni v_nj tau_n (-df/de)
            Hall: (e**3/V) sum_n (v_ni v_ni m_njj - v_ni v_nj m_nij) tau_n (-df/de)
            sigmaS: (e/V)/T  sum_n v_ni v_nj tau_n (-df/de) (e-mu)
            kappa: (1/V)/T  sum_n v_ni v_nj tau_n (-df/de) (e-mu)**2
        """
        self.mu_range = mu_range
        self.tmpr_range = tmpr_range
        self.tau_n = np.zeros([self.ws.num_wann], dtype=np.float)
        self.tau_n += tau
        result = self.parallel_k.apply_func_ksum(
            self.calc_boltz_k, 
            #np.zeros((4, 3, 3, len(self.mu_range), len(self.tmpr_range)), dtype=np.float64), 
        )
        sigma, sigmaS, kappa, Hall = result[:, 0], result[:, 1], result[:, 2], result[:, 3]

        # convert to SI unit, i.e., S/m
        factor = elementary_charge**3/hbar**2/self.ws.volume * 10**(-5)
        sigma = sigma/self.mp_points.nk*factor

        # convert to SI unit, i.e., A/m/K
        sigmaS = sigmaS/self.mp_points.nk*factor

        # convert to SI unit, i.e., W/m/K
        kappa = kappa/self.mp_points.nk*factor

        # convert to SI unit, i.e., 
        factor2 = elementary_charge**5/hbar**4/self.ws.volume * 10**(-40)
        Hall = Hall/self.mp_points.nk*factor2

        # store data
        self.sigma = sigma
        self.sigmaS = sigmaS
        self.kappa = kappa
        self.Hall = Hall

    def calc_boltz_k(self, k: np.ndarray) -> np.ndarray:
        if self.ws.use_ws_dist:
            ham_k = self.ws.calc_ham_k(k, rc=self.ws.nnkp.nw2r, diagonalize=True)
        else:
            ham_k = self.ws.calc_ham_k(k, diagonalize=True)
        # -df/de
        mdf = ham_k.get_minus_d_fermi(self.mu_range, self.tmpr_range)
        vkh = np.einsum("ai,ab->bi", ham_k.va, self.ws.a, optimize=True)
        mkh = np.einsum("abi,ac,bd->cdi", ham_k.ma, self.ws.a, self.ws.a, optimize=True)
        vvmh = np.einsum("an,an,bbn->abn", vkh, vkh, mkh, optimize=True) \
              - np.einsum("an,bn,abn->abn", vkh, vkh, mkh, optimize=True)
        vij = np.einsum("an,bn,n->abn", vkh, vkh, self.tau_n, optimize=True)
        # (ek - mu)
        e_mu = ham_k.ek[:, None] - self.mu_range[None, :]
        # (ek - mu)/tmpr
        e_mu_t = (ham_k.ek[:, None, None] - self.mu_range[None, :, None])\
                 /self.tmpr_range[None, None,:]

        # a,b: x,y,z,  n: band index,  i: mu_range,  j: tmpr_range
        sigma  = np.einsum("abn,nij->abij", vij.real, mdf, optimize=True)
        Hall = np.einsum("abn,n,nij->abij", vvmh.real, self.tau_n, mdf, optimize=True)
        sigmaS = np.einsum("abn,nij,nij->abij", vij.real, mdf, e_mu_t, optimize=True)
        kappa = np.einsum("abn,nij,ni,nij->abij", vij.real, mdf, e_mu, e_mu_t, optimize=True)

        return sigma, sigmaS, kappa, Hall

    def save(self, prefix: str):
        with open(prefix + "_sigma.dat", mode="w") as fp:
            self._save(fp, self.sigma)
        with open(prefix + "_sigmaS.dat", mode="w") as fp:
            self._save(fp, self.sigmaS)
        with open(prefix + "_kappa.dat", mode="w") as fp:
            self._save(fp, self.kappa)
        with open(prefix + "_Hall.dat", mode="w") as fp:
            self._save(fp, self.Hall)

    def _save(self, fp: TextIOWrapper, v: np.ndarray):
        for i, m in enumerate(self.mu_range):
            fp.write("\n")
            for j, T in enumerate(self.tmpr_range):
                fp.write(
                    ("{:15.6f}"*2 + "{:20.8f}"*6 + "\n").format(
                        m, T, 
                        v[0,0,i,j], v[0,1,i,j], v[1,1,i,j],
                        v[0,2,i,j], v[1,2,i,j], v[2,2,i,j],
                    )
                )


if __name__ == "__main__":
    main()
