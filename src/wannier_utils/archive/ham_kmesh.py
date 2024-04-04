#!/usr/bin/env python

import numpy as np
import multiprocessing
import itertools

class HamKmesh(object):
    """
      wannier system with uniform kmesh
      input:
        ws: Wannier_system
        kmesh: [nkx, nky, nkz]
        with_velocity: If True, calculate vk_list

      self.ws
      self.kmesh
      self.klist: numpy array of k-vector [ [0,0,0], [0.2,0,0], ... ]
      self.hk_list: list of hk (wannier gauge)
      self.vk_list: list of vk (wannier gauge)
    """

    def __init__(self, ws, kmesh, use_sym = False, magmoms = None, nproc = 1):
        self.ws = ws
        self.nproc = nproc
        self.use_sym = use_sym
        if use_sym:
            self.ws.set_sym(magmoms)
            self.sym_ops = self.ws.ms_ops

        self.gen_kpoints(np.array(kmesh))


    def _return_ik(self, k):
        ik = np.mod( np.rint((k[:]+2) * self.kmesh[:] - self.kshift[:] * 0.5), self.kmesh[:] )
        return np.int32(ik[2] + ik[1] * self.kmesh[2] + ik[0] * self.kmesh[1] * self.kmesh[2])

    def gen_kpoints(self, kmesh):
        if len(kmesh) == 6:
            self.kmesh = kmesh[0:3]
            self.kshift = kmesh[3:6]
        elif len(kmesh) == 3:
            self.kmesh = kmesh
            self.kshift = np.zeros(3)
        else:
            raise Exception("kmesh dimension mismatch")
        self.nk = np.prod(self.kmesh[0:3])

        kxyz = [ (np.arange(l)+s*0.5) / float(l) for l,s in zip(self.kmesh, self.kshift)]
        self.full_klist = np.array(list(itertools.product(kxyz[0], kxyz[1], kxyz[2])))
        print("# num of full klist: {}".format(len(self.full_klist)))
        for ik, k in enumerate(self.full_klist):
            assert ik == self._return_ik(k)

        if self.use_sym:
            self.irr_klist = self.irreducible_klist()
            print("# num of irr klist: {}".format(len(self.irr_klist)))

    def _klist(self, use_sym):
        if use_sym:
            return self.irr_klist
        else:
            return self.full_klist

    def gen_hk_list(self, use_sym = None):
        if use_sym is None:
            use_sym = self.use_sym
        klist = self._klist(use_sym)
        if self.nproc == 1:
            self.hk_list = [ self.ws.calc_hk(k) for k in klist ]
        else:
            with multiprocessing.Pool(self.nproc) as p:
                self.hk_list = p.map(self.ws.calc_hk, klist)

    def apply_func_k(self, func, recover_func=None):
        """
        return [ func(k) for k in self.klist ]
        """
        if self.nproc == 1:
            result = [ func(k) for k in self._klist(self.use_sym) ]
        else:
            with multiprocessing.Pool(self.nproc) as p:
                result = p.map(func, self._klist(self.use_sym))

        if self.use_sym and (recover_func is not None):
            full_result = []
            for ik, k in enumerate(self.full_klist):
                r = result[ self.equiv_k[ik] ]
                op = self.sym_ops[ self.equiv_sym[ik] ]
                full_result.append( recover_func(r, op) )
            result = full_result

        return result

    def _apply_func_ksum(self, ik_list):
        result_all = np.zeros_like(self._result0)
        use_sym = self.use_sym and (self._func_recover is not None)
        klist = self._klist(use_sym)
        for ik in ik_list:
            k = klist[ik]
            result = self._func_calc_k(k)
            if use_sym:
                for ikf in np.where(self.equiv_k == ik)[0]:
                    op = self.sym_ops[ self.equiv_sym[ikf] ]
                    result_all += self._func_recover(result, op)
            else:
                result_all += result
        return result_all

    def apply_func_ksum(self, result0, func_calc_k, func_recover=None):
        self._result0 = result0
        self._func_calc_k = func_calc_k
        self._func_recover = func_recover
        klist = self._klist(self.use_sym and (self._func_recover is not None))
        ik_list = np.arange( len(klist) )
        if self.nproc == 1:
            result = self._apply_func_ksum(ik_list)
        else:
            with multiprocessing.Pool(self.nproc) as p:
                result = p.map(self._apply_func_ksum, np.array_split(ik_list, self.nproc))
            result = np.sum(result, axis=0)
        return result

    def irreducible_klist(self):
        irr_klist = []
        equiv_k = - np.ones([self.nk], dtype=int)
        equiv_sym = np.zeros([self.nk], dtype=int)
        iks = 0
        for ik, k in enumerate(self.full_klist):
            if equiv_k[ik] >= 0: continue
            equiv_k[ik] = iks
            equiv_sym[ik] = 0  # iop = 0 corresponds to identity
            irr_klist.append(k)
            for iop, op in enumerate(self.sym_ops):
                kn = k @ op.rotation_matrix
                if hasattr(op, "time_reversal"): kn *= op.time_reversal
                ikn = self._return_ik(kn)
                if equiv_k[ikn] < 0:
                    equiv_k[ikn] = iks
                    equiv_sym[ikn] = iop
            iks += 1
        self.equiv_k = equiv_k
        self.equiv_sym = equiv_sym
        #for ik in range(self.nk):
        #    print(ik, self.equiv_k[ik], self.equiv_sym[ik])
        return np.array(irr_klist)

    def check_sym(self, rtol=1e-12, atol = 0):
        """
        check symmetry of e_k and v_k
        for symmetry of v_k, use use_ws_dist.
        """
        self.gen_hk_list()
        for ik, k in enumerate(self.full_klist):
            hk = self.ws.calc_hk(k)
            iks = self.equiv_k[ik]
            iop = self.equiv_sym[ik]
            op = self.sym_ops[iop]
            ks = self.irr_klist[iks]
            #print("--comp ik = {}, iks = {}, isym = {} --".format(ik, iks, iop))
            #print(op.rotation_matrix)
            #print(op.time_reversal)
            #print(hk.e)
            err  = "ik  = {}, k = {} {} {}\n".format(ik, k[0], k[1], k[2])
            err += "iks = {}, k = {} {} {}".format(iks, ks[0], ks[1], ks[2])
            np.testing.assert_allclose(self.hk_list[iks].e, hk.e, rtol=rtol, atol=atol, err_msg=err)

            sgn = 1
            if hasattr(op, "time_reversal"): sgn *= op.time_reversal
            inv_mat = np.linalg.inv(op.rotation_matrix)
            v1 = hk.HHa[:,0,0]
            v2 = np.einsum("ba,a->b", inv_mat, self.hk_list[iks].HHa[:,0,0]) * sgn
            np.testing.assert_allclose(v1, v2, atol = 1e-10, rtol = rtol, err_msg=err)

