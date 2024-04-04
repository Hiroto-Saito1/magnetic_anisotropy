#!/usr/bin/env python

from typing import Iterable, Optional, Union

import numpy as np
from scipy.constants import Boltzmann, eV
from scipy.special import expit
from pymatgen.core.operations import SymmOp, MagSymmOp

from wannier_utils.hamiltonian import HamK
from wannier_utils.wannier_kmesh import WannierKmesh
from wannier_utils.wannier_system import WannierSystem
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


def main():
    from pathlib import Path
    from time import perf_counter

    base_dir = Path(__file__).parents[2]
    ws = WannierSystem(
        hr_dat=base_dir/"tests"/"Fe_hr.dat", 
        nnkp_file=base_dir/"tests"/"Fe.nnkp", 
        win_file=base_dir/"tests"/"Fe.win", 
    )

    start_time = perf_counter()
    kmesh = [10, 10, 10]
    ef = 17.4058
    mu_range = ef + np.arange(-0.2, 0.2, 0.02)
    tmpr_range = np.linspace(100, 500, 5, dtype=np.float64)
    berry = BerryCurvature(ws, kmesh, nproc=4)
    berry.calc_ahc_anc(mu_range, tmpr_range)
    berry.save("Fe")
    end_time = perf_counter()
    logger.info(f"elapsed_time: {end_time - start_time} [sec]")


class BerryCurvature(WannierKmesh):
    """
    Berry curvature calculation.
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

    def calc_ahc_anc(self, mu_range: np.ndarray, tmpr_range: np.ndarray):
        """
        Args:
            mu_range (np.ndarray): chemical potentials in unit of eV.
            tmpr_range (np.ndarray): temperatures in unit of K.
        """
        self.mu_range = np.array(mu_range)
        self.tmpr_range = np.array(tmpr_range)
        result = self.parallel_k.apply_func_ksum(
            self.calc_ahc_k, 
            #np.zeros((2, 3, 3, len(self.mu_range), len(self.tmpr_range)), dtype=np.float64), 
            func_recover=self.recover_ahc, 
        )
        result = np.einsum(
            "ac,bd,iabmt->icdmt", 
            self.ws.a, self.ws.a, result, 
            optimize=True, 
        )
        ahc, anc = result[0], result[1]

        # e^{2}/hbar/V
        # h/e^{2} = 25812.807...: quantum Hall constant
        # convert unit to S/cm
        factor = 2*np.pi/25812.8074434/self.ws.volume * 10**8
        # convert unit to A/Km
        factor2 = factor*0.008617
        self.ahc = ahc*factor/np.prod(self.mp_points.nk)
        self.anc = anc*factor2/np.prod(self.mp_points.nk)

    def calc_ahc_k(self, k: np.ndarray) -> np.ndarray:
        if self.ws.use_ws_dist:
            ham_k = self.ws.calc_ham_k(k, rc=self.ws.nnkp.nw2r, diagonalize=True)
        else:
            ham_k = self.ws.calc_ham_k(k, diagonalize=True)
        f = ham_k.get_fermi(self.mu_range, self.tmpr_range)
        g = self.get_fermi_ane(ham_k.ek, f, self.mu_range, self.tmpr_range)
        berry = self.calc_berry(ham_k)
        #ahc = np.einsum("nmt,abn->abmt", f, berry.real)
        #anc = np.einsum("nmt,abn->abmt", g, berry.real)
        res = np.einsum("pnmt,abn->pabmt", f, g, berry.real, optimize=True)
        return res

    def calc_berry(self, ham_k: HamK) -> np.ndarray:
        dd = np.einsum("aij,bji->abi", ham_k.DHa, ham_k.DHa, optimize=True)
        return 1j*(dd - dd.transpose((1, 0, 2)))

    def recover_ahc(self, res: np.ndarray, op: Union[SymmOp, MagSymmOp]):
        #ahc, anc = res
        mat = op.rotation_matrix
        inv_mat = np.linalg.inv(mat)
        sgn = op.time_reversal if hasattr(op, "time_reversal") else 1
        #ahc_new = np.einsum("ca,db,abmt->cdmt", inv_mat, inv_mat, ahc, optimize=True)*sgn
        #anc_new = np.einsum("ca,db,abmt->cdmt", inv_mat, inv_mat, anc, optimize=True)*sgn
        res_new = np.einsum("ca,db,pabmt->pcdmt", inv_mat, inv_mat, res, optimize=True)*sgn
        return res_new

    def get_fermi_ane(
        self, 
        e: np.ndarray, 
        f: np.ndarray, 
        mu_range: np.ndarray, 
        tmpr_range: np.ndarray, 
    ) -> np.ndarray:
        """
        mu_range (np.ndarray): chemical potentials in unit of eV.
        tmpr_range (np.ndarray): temperature in unit of K.
        """
        tmpr_range_eV = tmpr_range*Boltzmann/eV
        e_mu = e[:, None, None] - mu_range[None, :, None]
        t = tmpr_range_eV[None, None, :]
        x = e_mu/t
        g = x*f
        #return np.where(x < -50, g - x, g + np.log(1 + np.exp(-x)))
        a = np.where(x < -50, 1, expit(x))
        return np.where(x < -50, g - x, g - np.log(a))

    def save(self, prefix: str):
        with open(prefix + "_ahc.dat", mode="w") as fp:
            for j, tmpr in enumerate(self.tmpr_range):
                fp.write("\n")
                for i, m in enumerate(self.mu_range):
                    fp.write(
                        ("{:15.6f}"*2 + "{:20.6f}"*6 + "\n").format(
                            m, tmpr, 
                            self.ahc[1,2,i,j], self.ahc[2,0,i,j], self.ahc[0,1,i,j], 
                            self.anc[1,2,i,j], self.anc[2,0,i,j], self.anc[0,1,i,j], 
                        )
                    )


if __name__ == "__main__":
    main()
