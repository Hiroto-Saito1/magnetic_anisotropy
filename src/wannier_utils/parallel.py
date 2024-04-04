#!/usr/bin/env python

from multiprocessing import Pool
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

from wannier_utils.wannier_system import WannierSystem
from wannier_utils.hamiltonian import HamK
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


class ParallelK:
    """
    Parallelization of methods in reciprocal space.
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        kvec: np.ndarray, 
        is_irreducible: bool = False, 
        equiv_k: Optional[np.ndarray] = None, 
        equiv_sym: Optional[np.ndarray] = None, 
        nproc: int = 1, 
    ):
        self.ws = ws
        self.kvec = kvec
        self.is_irreducible = is_irreducible
        if self.is_irreducible:
            assert hasattr(self.ws, "sym_ops"), "WannierSystem does not have symmetry operations."
        self.equiv_k = equiv_k
        self.equiv_sym = equiv_sym
        self.nproc = nproc

    def get_ham_ks(self, rc: Optional[np.ndarray] = None, diagonalize: bool = False) \
    -> List[HamK]:
        return self.apply_func_k(self.ws.calc_ham_k, args=(rc, diagonalize))

    def apply_func_k(
        self, 
        func_k: Callable, 
        args: Optional[Iterable[Any]] = None, 
        func_recover: Optional[Callable] = None, 
    ) -> List[Any]:
        """
        Return [func_k(k, *args) for k in kvec] by parallelization.

        Args:
            func_k (Callable): a method whose argument is k.
            args (Optional[Iterable[Any]]): additional arguments of func_k. defaults to None.
            func_recover (Optional[Callable]): 
                function recovering the results at irreducible k points, whose
                arguments should be (result_k, sym_op).

        Returns:
            List[Any]: [func_k(k, *args) for k in kvec].
        """
        if self.nproc == 1:
            results = [func_k(k) if args is None else func_k(k, *args) for k in self.kvec]
        else:
            args_k = self.kvec if args is None else [[k, *args] for k in self.kvec]
            with Pool(self.nproc) as p:
                func_map = p.map if args is None else p.starmap
                results = func_map(func_k, args_k)

        if not self.is_irreducible:
            return results
        assert func_recover is not None, "func_recover is required for irreducible k points."
        results_all = [None]*len(self.equiv_k)
        for ik, result in enumerate(results):
            for ikr in np.where(self.equiv_k == ik)[0]:
                sym_op = self.ws.sym_ops[self.equiv_sym[ikr]]
                results_all[ikr] = func_recover(result, sym_op)

        assert not None in results_all, "Failed to map results of irreducible k points."
        return results_all

    def apply_func_ksum(
        self, 
        func_k: Callable,
        #result0: np.ndarray, 
        args: Optional[Iterable] = None, 
        func_recover: Optional[Callable] = None, 
    ) -> Any:
        """
        Return sum([func_k(k, *args) for k in kvec]) by parallelization.
        To restrain memory consumption, verbose for loop is implemented at summation.

        Args:
            func_k (Callable): a method whose argument is k.
            result0 (np.ndarray): zero-filled array in shape of summation result.
            args (Optional[Iterable]): additional arguments of func_k. defaults to None.
            func_recover (Optional[Callable]): 
                function recovering the results at irreducible k points, whose
                arguments should be (result_k, sym_op).

        Returns:
            Any: sum([func_k(k, *args) for k in kvec]).

        Note:
            If func_k returns a scalar, it is better to use sum(self.apply_func_k()).
        """
        self._func_k = func_k
        #self._result0 = result0
        self._args = args
        self._func_recover = func_recover

        inds_kvec_all = np.arange(len(self.kvec))
        if self.nproc == 1:
            return self._exec_ksum(inds_kvec_all)
        with Pool(self.nproc) as p:
            results = p.map(self._exec_ksum, np.array_split(inds_kvec_all, self.nproc))
        if isinstance(results[0], (list, tuple)):
            result = results[0]
            for i in range(1, self.nproc):
                result = [r + ri for r, ri in zip(result, results[i])]
        else:
            result = np.sum(results, axis=0)
        return result

    def _exec_ksum(self, inds_kvec: np.ndarray) -> Any:
        """
        Internal method of self.apply_func_ksum.
        This function is separated form self.apply_func_ksum for parallelization.

        Args:
            inds_kvec (np.ndarray): indices of kvec.

        Returns:
            Any: result of summation over given k points.
        """
        #results = np.zeros_like(self._result0)
        result = None
        for ik in inds_kvec:
            if self._args is None:
                result_new = self._func_k(self.kvec[ik])
            else:
                result_new = self._func_k(self.kvec[ik], *self._args)
            if self.is_irreducible:
                assert self._func_recover is not None, "func_recover is required for irreducible k points."
                for ikr in np.where(self.equiv_k == ik)[0]:
                    op = self.ws.sym_ops[self.equiv_sym[ikr]]
                    #results += self._func_recover(result, op)
                    result_new = self._func_recover(result_new, op)
            #else:
            #    results += result
            if result is None:
                result = result_new
                continue
            if isinstance(result_new, (list, tuple)):
                result = [r + rn for r, rn in zip(result, result_new)]
            else:
                result += result_new
        return result
