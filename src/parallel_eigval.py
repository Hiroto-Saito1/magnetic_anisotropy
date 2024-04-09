# -*- coding: utf-8 -*-

"""与えられたkメッシュ上でのハミルトニアンの対角化をmpi並列化するモジュール。

Example:
    >>> mpirun -n 8 python parallel_eigval.py
"""

import numpy as np
from mpi4py import MPI
from typing import Optional
from pathlib import Path
import itertools
import os

from mag_rotation import MagRotation
from wannier_utils.hamiltonian import HamR, HamK


class ParallelEigval:
    """
    与えられたk点リスト上でハミルトニアンの対角化を並列化し、エネルギー固有値を返すクラス。

    Args:
        ham_r (HamR or MagRotation): Real space Hamiltonian object.
        klist: List of kpoints in (klen, 3) array.

    Attributes:
        comm (MPI.Comm): MPI通信オブジェクト。
        rank (int): 現在のプロセスのランク。
        nproc (int, optional): 使用するプロセスの総数。
        energy: List of eigvals in (num_wann, klen) array.
    """

    def __init__(self, ham_r: HamR, klist: np.ndarray):
        self.energy = self.scatter(ham_r, klist)

    def scatter(
        self, ham_r: HamR, klist: np.ndarray,
    ) -> np.ndarray:
        """対角化を分散する。

        Args:
            ham_r (HamR or MagRotation): Real space Hamiltonian object.
            klist: List of kpoints in (klen, 3) array.

        Returns:
            eigval: List of eigvals in (num_wann, klen) array.
        """
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nproc = self.comm.Get_size()

        if self.rank == 0:
            _k_assigned = np.array_split(klist, self.nproc)
        else:
            _k_assigned = None

        _k_assigned = self.comm.scatter(_k_assigned, root=0)
        _eigval_assigned = self.diag_ham(ham_r, _k_assigned)
        eigval_list = self.comm.gather(_eigval_assigned, root=0)
        if self.rank == 0:
            eigval = np.concatenate(eigval_list, axis=0)
            return eigval.reshape(-1, ham_r.num_wann).T

    def diag_ham(self, ham_r: HamR, klist: np.ndarray) -> np.ndarray:
        eigval = np.zeros((len(klist), ham_r.num_wann), dtype=np.float_)
        for i, k in enumerate(klist):
            ham_k = HamK(ham_r, k, diagonalize=True)
            eigval[i] = ham_k.ek
        return eigval


if __name__ == "__main__":

    def get_klist(nk1, nk2, nk3):
        k123 = [np.arange(nk) / float(nk) for nk in [nk1, nk2, nk3]]
        klist = list(itertools.product(k123[0], k123[1], k123[2]))
        return np.array(klist)

    tests_path = Path(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))
    )
    tb_dat = tests_path / "mp-2260/pwscf_rel_sym/mae/wan/pwscf_tb.dat"
    ham = MagRotation(tb_dat=tb_dat)
    parallel = ParallelEigval(ham, get_klist(5, 5, 5))
    
    if np.any(parallel.energy != None):
        print(type(parallel.energy), np.shape(parallel.energy))
        print(f"rank: {parallel.rank}\n")
        print(f"Number of processes: {parallel.nproc} \n")  

