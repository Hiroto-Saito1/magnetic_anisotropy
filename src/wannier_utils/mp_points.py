#!/usr/bin/env python

from itertools import product
from typing import Iterable, Optional, Tuple, Union

import numpy as np

from wannier_utils.wannier_system import WannierSystem


class MPPoints:
    """
    WannierSystem with Monkhorst-Pack k points.
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        kmesh: Iterable[int], 
        magmoms: Optional[Union[str, np.ndarray]] = None, 
        use_sym: bool = False, 
    ):
        """
        Constructor.

        Args:
            ws (WannierSystem): _description_
            kmesh (Iterable[int]): _description_
            magmoms (Optional[Union[str, np.ndarray]], optional): _description_. defaults to None.
            use_sym (bool, optional): _description_. defaults to False.
        """
        self.kmesh = kmesh[:3]
        if len(kmesh) == 6:
            self.kshift = kmesh[3:]
        elif len(kmesh) == 3:
            self.kshift = (0, 0, 0)
        else:
            raise ValueError("kmesh dimension mismatch.")

        self.ws = ws
        self.ws.set_sym(magmoms=magmoms)
        self.nk = np.prod(self.kmesh)
        self.full_kvec = self._get_full_kvec()
        self.use_sym = use_sym
        if self.use_sym:
            self.irr_kvec, self.equiv_k, self.equiv_sym = self._get_irr_kvec()
        else:
            self.irr_kvec, self.equiv_k, self.equiv_sym = None, None, None

    def _get_full_kvec(self) -> np.ndarray:
        k123 = [(np.arange(l) + s*0.5)/float(l) for l, s in zip(self.kmesh, self.kshift)]
        full_kvec = np.array(list(product(*k123)))
        for ik, k in enumerate(full_kvec):
            assert ik == self._get_ik(k), "Mismatch of index of k point."
        return full_kvec

    def _get_ik(self, k: np.ndarray) -> int:
        ik = np.mod(
            np.rint((k + 2)*np.array(self.kmesh[:]) - np.array(self.kshift)*0.5), 
            np.array(self.kmesh), 
        )
        return int(ik[2] + ik[1]*self.kmesh[2] + ik[0]*self.kmesh[1]*self.kmesh[2])

    def _get_irr_kvec(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return irreducible k points.

        Returns:
            irr_kvec (np.ndarray): irreducible k points.
            equiv_k (np.ndarray): label of equivalent k points.
            equiv_sym (np.ndarray): index of symmetry operation to make each k point equivalent with the irreducible k point.
        """
        irr_kvec = []
        equiv_k = -np.ones([self.nk], dtype=np.int64)
        equiv_sym = np.zeros([self.nk], dtype=np.int64)
        iks = 0
        for ik, k in enumerate(self.full_kvec):
            if equiv_k[ik] >= 0: continue
            irr_kvec.append(k)
            equiv_k[ik] = iks
            equiv_sym[ik] = 0   # iop = 0 corresponds to identity
            for iop, op in enumerate(self.ws.sym_ops):
                kn = (k@op.rotation_matrix)
                if hasattr(op, "time_reversal"): kn *= op.time_reversal
                ikn = self._get_ik(kn)
                if equiv_k[ikn] < 0:
                    equiv_k[ikn] = iks
                    equiv_sym[ikn] = iop
            iks += 1
        return np.array(irr_kvec), equiv_k, equiv_sym

    def check_sym(
        self, 
        rtol: float = 1e-12, 
        atol: float = 0, 
        rc: Optional[np.ndarray] = None, 
        nproc: int = 1, 
    ):
        """
        Check symmetry of e_k and u_k (for symmetry of u_k, use use_ws_dist).
        """
        from wannier_utils.parallel import ParallelK

        if not self.use_sym:
            raise ValueError("use_sym must be True.")
        ham_ks = ParallelK(
            self.ws, 
            self.irr_kvec, 
            is_irreducible=True, 
            equiv_k=self.equiv_k, 
            equiv_sym=self.equiv_sym, 
            nproc=nproc, 
        ).get_ham_ks(rc=rc)
        for ik, k in enumerate(self.full_kvec):
            ham_k = self.ws.calc_ham_k(k, rc=rc)
            iks = self.equiv_k[ik]
            iop = self.equiv_sym[ik]
            op = self.ws.sym_ops[iop]
            ks = self.irr_kvec[iks]
            #print("-- Compare ik = {}, iks = {}, isym = {} --".format(ik, iks, iop))
            #print(op.rotation_matrix)
            #print(op.time_reversal)
            #print(ham_k.ek)
            err  = "ik  = {}, k = {} {} {}\n".format(ik, k[0], k[1], k[2])
            err += "iks = {}, k = {} {} {}".format(iks, ks[0], ks[1], ks[2])
            np.testing.assert_allclose(ham_ks[iks].ek, ham_k.ek, rtol=rtol, atol=atol, err_msg=err)

            sgn = op.time_reversal if hasattr(op, "time_reversal") else 1
            inv_mat = np.linalg.inv(op.rotation_matrix)
            v1 = ham_k.HHa[:,0,0]
            v2 = np.einsum("ba,a->b", inv_mat, ham_ks[iks].HHa[:, 0, 0])*sgn
            np.testing.assert_allclose(v1, v2, rtol=rtol, atol=1e-10, err_msg=err)
