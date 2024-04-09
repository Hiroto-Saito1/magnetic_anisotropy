# -*- coding: utf-8 -*-

"""与えられたkメッシュ上でのハミルトニアンの対角化をmpi並列化するモジュール。

Example:
    >>> mpirun -n 8 python parallel_eigval.py
"""

import time
import os
import psutil
import itertools
import numpy as np
from mpi4py import MPI
from pathlib import Path

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
        nproc (int): 使用するプロセスの総数。
        energy: List of eigvals in (num_wann, klen) array.

    Notes:
        klen は nporc の倍数である必要がある。
    """

    def __init__(self, ham_r: HamR, klist: np.ndarray):
        self.energy = self.scatter(ham_r, klist)

    def scatter(self, ham_r: HamR, klist: np.ndarray) -> np.ndarray:
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
            eigval = np.empty(
                (self.nproc, len(_k_assigned[0]), ham_r.num_wann), dtype=float
            )
        else:
            _k_assigned = None
            eigval = None
        _k_assigned = self.comm.scatter(_k_assigned, root=0)
        _eigval_assigned = self.diag_ham(ham_r, _k_assigned)
        self.comm.Gather(_eigval_assigned, eigval, root=0)
        if self.rank == 0:
            return eigval.reshape(-1, ham_r.num_wann).T

    def diag_ham(self, ham_r: HamR, klist: np.ndarray) -> np.ndarray:
        eigval = np.zeros((len(klist), ham_r.num_wann), dtype=np.float_)
        for i, k in enumerate(klist):
            ham_k = HamK(ham_r, k, diagonalize=True)
            eigval[i] = ham_k.ek
        return eigval


if __name__ == "__main__":
    """テストケース用。"""

    def get_klist(nk1, nk2, nk3):
        k123 = [np.arange(nk) / float(nk) for nk in [nk1, nk2, nk3]]
        klist = list(itertools.product(k123[0], k123[1], k123[2]))
        return np.array(klist)

    t_start = time.time()
    Nk = 10
    klist = get_klist(Nk, Nk, Nk)
    ham_r = HamR(hr_dat=Path("../tests/mp-2260/pwscf_rel_sym/mae/wan/pwscf_py_hr.dat"))
    parallel = ParallelEigval(ham_r, klist)
    print(parallel.nproc)
    print(type(parallel.energy), np.shape(parallel.energy))
    if parallel.rank == 0:
        # print(np.shape(parallel.energy))
        walltime = time.time() - t_start
        print(f"fin: {walltime:.3f} sec")

        # 現在のプロセスIDを取得
        process_id = os.getpid()
        # プロセスのメモリ情報を取得
        process_info = psutil.Process(process_id)
        # メモリ使用量を取得
        memory_use = process_info.memory_info().rss / 1024 / 1024 / 1024  # 単位はGB
        # 結果を出力
        print(f"Memory used by the program: {memory_use} GB")
