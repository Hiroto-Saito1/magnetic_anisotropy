#!/usr/bin/env python

import sys
import os
import numpy as np
import scipy
import scipy.constants
import itertools
import tomli
from wannier_utils.ham_kmesh import HamKmesh
from wannier_utils.temperature_ir2 import Temperature

kelvin2eV = scipy.constants.Boltzmann/scipy.constants.eV

#< S > = sigma_s1s2 < c_s1^dagger c_s2 > = - sigma_s1s2 G(i s2,i s1, tau=+0)   (### see AGD (7.1) and after that)

class Jij(HamKmesh):
    def __init__(self, ws, kmesh, mu, temperature_in_K, axis='x', nproc = 1):
        super().__init__(ws, kmesh, nproc = nproc)
        self.mu = mu
        self.T = Temperature(temperature_in_K)
        self.dim = self.ws.num_wann
        self.dim2 = self.dim//2

        self.up_state = np.zeros([self.dim, self.dim2], dtype=complex)
        self.dn_state = np.zeros([self.dim, self.dim2], dtype=complex)
        if 'z' in axis:
            print(" spin axis = z")
            for i in range(self.dim2):
                self.up_state[2*i:2*i+2,i] = np.array([1, 0])
                self.dn_state[2*i:2*i+2,i] = np.array([0, 1])
        elif 'x' in axis:
            print(" spin axis = x")
            for i in range(self.dim2):
                self.up_state[2*i:2*i+2,i] = np.array([1,-1])/np.sqrt(2)
                self.dn_state[2*i:2*i+2,i] = np.array([1, 1])/np.sqrt(2)
        else:
            raise Exception("invalid axis")

        print ('  Jij.py: calculate J from Green function')
        print ('  mu (eV)       = {:12.6f}'.format(self.mu))
        print ('  T  (K )       = {:12.2f}'.format(temperature_in_K))
        print ('  (nk1,nk2,nk3) = ({0[0]:d}, {0[1]:d}, {0[2]:d})'.format(self.kmesh))
        print ('  natom         = {:d}'.format(self.ws.natom))
        print (' ')
        print ('  atom_pos')

        for i in range(len(self.ws.nnkp.atom_pos)):
            print('  %2d: %12.6f %12.6f %12.6f' % ((i,) + tuple(self.ws.nnkp.atom_pos_r[i])))

    def calc_rlist(self, rcut):
        rl = range(-5,6)
        r_list_all = np.array([ [rx,ry,rz] for rx, ry, rz in itertools.product(rl, rl, rl)])
        r_list = []
        for r in r_list_all:
            if np.linalg.norm(np.einsum("ba,b->a", self.ws.a, r)) < rcut:
                r_list.append(r)
        return np.array(r_list)

    def calc_gn_wan(self, hk):
        """
        calculate G_iwn_sp in wannier gauge
        """
        ek, uk = scipy.linalg.eigh(hk)
        g0iwn_sp = 1 / (1j * self.T.omegaF_sp[:, np.newaxis] - (ek[np.newaxis, :] - self.mu))
        return np.einsum("mn, ln, on-> lmo", uk, g0iwn_sp, np.conj(uk), optimize='optimal')

    def calc_proj_delta(self):
        onsite_mat = self.ws.ham_r.onsite_mat()  # num_wann x num_wann matrix
        sigma_up = np.einsum("im,ij,jn->mn", self.up_state, onsite_mat, self.up_state, optimize='optimal')
        sigma_dn = np.einsum("im,ij,jn->mn", self.dn_state, onsite_mat, self.dn_state, optimize='optimal')
        delta = (sigma_up - sigma_dn)/2

        self.orb_index = []
        self.delta_list = []
        for pos in self.ws.nnkp.atom_pos:
            orb_ind = np.array( sum([ self.ws.nnkp.atom_orb[orb] for orb in pos ], []) )
            orb_ind = orb_ind[::2] // 2
            self.orb_index.append( orb_ind )
            proj_delta = np.take(np.take(delta, orb_ind, axis=0), orb_ind, axis=1)
            self.delta_list.append( proj_delta )

    def calc_internal(self, k):
        hk = self.ws.calc_hk(k, diagonalize=False)
        hk_up = np.einsum("im,ij,jn->mn", self.up_state, hk.mat, self.up_state, optimize='optimal')
        hk_dn = np.einsum("im,ij,jn->mn", self.dn_state, hk.mat, self.dn_state, optimize='optimal')
        gnw_up = self.calc_gn_wan(hk_up)
        gnw_dn = self.calc_gn_wan(hk_dn)

        Jij_sum = np.zeros([self.ws.natom, self.ws.natom])
        for na, nb in itertools.product(range(self.ws.natom), range(self.ws.natom)):
            gnw_up_ab = np.take(np.take(gnw_up, self.orb_index[na], axis=1), self.orb_index[nb], axis=2)
            gnw_dn_ba = np.take(np.take(gnw_dn, self.orb_index[nb], axis=1), self.orb_index[na], axis=2)
            Jij_sum[na,nb] = -np.einsum("n,ij,njk,km,nmi->", self.T.sum_factor, self.delta_list[na], gnw_up_ab, self.delta_list[nb], gnw_dn_ba, optimize='optimal').real
        return [gnw_up, gnw_dn, Jij_sum]

    def calc_all(self, rcut):
        # calc delta_list, orb_index
        self.calc_proj_delta()

        # calc for each k points
        self.result = self.apply_func_k(self.calc_internal)

        self.calc_Jq0()  # q=0 mode
        if rcut == 0:
            return
        elif rcut > 0: # r list mode
            self.calc_Jrlist(rcut)
        else:          # all q mode
            self.calc_Jq()

    def trace_sgsg(self, gu, gd):
        Jij = np.zeros([self.ws.natom, self.ws.natom])
        for na, nb in itertools.product(range(self.ws.natom), range(self.ws.natom)):
            gu_ab = np.take(np.take(gu, self.orb_index[na], axis=1), self.orb_index[nb], axis=2)
            gd_ba = np.take(np.take(gd, self.orb_index[nb], axis=1), self.orb_index[na], axis=2)
            Jij[na,nb] = -np.einsum("n,ij,njk,km,nmi->", self.T.sum_factor, self.delta_list[na], gu_ab, self.delta_list[nb], gd_ba, optimize='optimal').real
        return Jij

    def calc_Jq0(self):
        gnw_up_kall = [ gnw_up_k for gnw_up_k, _, _ in self.result ]
        gnw_dn_kall = [ gnw_dn_k for _, gnw_dn_k, _ in self.result ]
        Jij_sum = np.sum([ Jij_k for _, _, Jij_k in self.result ], axis=0)/self.nk

        gnw_up_ksum = np.sum(gnw_up_kall, axis=0)/self.nk
        gnw_dn_ksum = np.sum(gnw_dn_kall, axis=0)/self.nk

        nup = np.einsum("n,nmm->", self.T.sum_factor, gnw_up_ksum, optimize='optimal').real
        ndn = np.einsum("n,nmm->", self.T.sum_factor, gnw_dn_ksum, optimize='optimal').real
        ne = nup + ndn + self.ws.num_wann * 0.5
        ns = nup - ndn
        print("")
        print("  number of electrons: {:12.5f}".format(ne))
        print("  total spin:          {:12.5f}".format(ns))
        print("")

        # Jij (i,j in R=0)
        Jij = self.trace_sgsg(gnw_up_ksum, gnw_dn_ksum)

        # Jij(q=0)
        Jij_q0 = Jij_sum
        for na in range(self.ws.natom):
            Jij_q0[na,na] -= Jij[na,na]

        print("  J0_A (d^2F / d theta^2)")
        Ji = np.zeros([self.ws.natom])
        for na in range(self.ws.natom):
            gnw_up = np.take(np.take(gnw_up_ksum, self.orb_index[na], axis=1), self.orb_index[na], axis=2)
            gnw_dn = np.take(np.take(gnw_dn_ksum, self.orb_index[na], axis=1), self.orb_index[na], axis=2)
            Ji[na] = np.einsum("n,ij,nji", self.T.sum_factor2, self.delta_list[na], (gnw_up - gnw_dn)/2, optimize='optimal').real
        J = np.sum( [ -Jij[na,na] - Ji[na] for na in range(self.ws.natom) ] ) / self.ws.natom
        print("  J0_A (eV):             {:12.5f}".format(J))
        print("  J0_A (K):              {:12.5f}".format(J/kelvin2eV))
        print("  Tc_A (K):              {:12.5f}".format(J/kelvin2eV*2/3))
        print("")
        self.JA = J  # for test

        print("  J0_B (sum Jij)")
        J = np.sum(Jij_q0)
        print("  J0_B (eV):             {:12.5f}".format(J))
        print("  J0_B (K):              {:12.5f}".format(J/kelvin2eV))
        print("  Tc_B (K):              {:12.5f}".format(J/kelvin2eV*2/3))
        print("")
        self.JB = J  # for test

        print("  Ji (eV)")
        for na in range(self.ws.natom):
            print("   {:5d}:     {:12.5f}".format(na, -Jij[na,na] - Ji[na]))
        print("")

        print("  J(q=0)")
        for na, nb in itertools.product(range(self.ws.natom), range(self.ws.natom)):
            print("    {:5d} {:5d}  {:12.5f}".format(na,nb,Jij_q0[na,nb]))
        np.save("Jij_q0", Jij_q0)

    def calc_Jrlist(self, rcut):
        gnw_up_kall = [ gnw_up_k for gnw_up_k, _, _ in self.result ]
        gnw_dn_kall = [ gnw_dn_k for _, gnw_dn_k, _ in self.result ]
        Jij_sum = np.sum([ Jij_k for _, _, Jij_k in self.result ], axis=0)/self.nk

        # r_list and exp(ikr)
        self.r_list = self.calc_rlist(rcut)
        nr = len(self.r_list)
        ir0 = np.where( (self.r_list[:,0] == 0) & (self.r_list[:,1] == 0) & (self.r_list[:,2] == 0) )[0][0]
        exp_kr = np.exp(1j*2*np.pi*np.einsum("ki,ri->kr", self.full_klist, self.r_list, optimize='optimal'))
        gnw_up_ksum = np.einsum("kr,knmp->rnmp", exp_kr, np.array(gnw_up_kall), optimize='optimal')
        gnw_dn_ksum = np.einsum("kr,knmp->rnmp", np.conj(exp_kr), np.array(gnw_dn_kall), optimize='optimal')
        gnw_up_ksum /= self.nk
        gnw_dn_ksum /= self.nk

        Jijr = np.zeros([nr, self.ws.natom, self.ws.natom])
        for ir in range(nr):
            Jijr[ir] = self.trace_sgsg(gnw_up_ksum[ir], gnw_dn_ksum[ir])

        print("  rcut:                {:12.5f}".format(rcut) )
        print("  num ij:              {:12d}".format(len(self.r_list)) )
        print("  sum Jij (eV):        {:12.5f}".format(np.sum(Jijr) - np.sum(np.diag(Jijr[ir0,:,:]))) )
        print("")
        print("  each Jij (eV):")
        for ir, r in enumerate(self.r_list):
            for na, nb in itertools.product(range(self.ws.natom), range(self.ws.natom)):
                print("   {:3d} {:3d} {:3d}: {:2d} {:2d}   {:12.5f}".format(r[0], r[1], r[2], na, nb, Jijr[ir,na,nb]))

    def calc_Jq(self):
        gnw_up_k = np.array([ gnw_up_k for gnw_up_k, _, _ in self.result ]).reshape(self.kmesh[0], self.kmesh[1], self.kmesh[2], len(self.T.omegaF_sp), self.dim2, self.dim2)
        gnw_dn_k = np.array([ gnw_dn_k for _, gnw_dn_k, _ in self.result ]).reshape(self.kmesh[0], self.kmesh[1], self.kmesh[2], len(self.T.omegaF_sp), self.dim2, self.dim2)
        gnw_up_r = np.fft.fftn(gnw_up_k, axes=(0,1,2))/self.nk
        gnw_dn_r = np.fft.ifftn(gnw_dn_k, axes=(0,1,2))
        Jijr = np.zeros([self.kmesh[0], self.kmesh[1], self.kmesh[2], self.ws.natom, self.ws.natom])
        print("Jij(r)")
        for rx, ry, rz in itertools.product(range(self.kmesh[0]), range(self.kmesh[1]), range(self.kmesh[2])):
            Jijr[rx,ry,rz] = self.trace_sgsg(gnw_up_r[rx,ry,rz], gnw_dn_r[rx,ry,rz])
            for na, nb in itertools.product(range(self.ws.natom), range(self.ws.natom)):
                print("   {:3d} {:3d} {:3d}: {:2d} {:2d}   {:12.5f}".format(rx, ry, rz, na, nb, Jijr[rx,ry,rz,na,nb]))
        print("Jij(q)")
        for na in range(self.ws.natom):
            Jijr[0,0,0,na,na] = 0
        Jijq = np.fft.fftn(Jijr, axes=(0,1,2))
        for rx, ry, rz in itertools.product(range(self.kmesh[0]), range(self.kmesh[1]), range(self.kmesh[2])):
            for na, nb in itertools.product(range(self.ws.natom), range(self.ws.natom)):
                print("   {:3d} {:3d} {:3d}: {:2d} {:2d}   {:12.5f}".format(rx, ry, rz, na, nb, Jijq[rx,ry,rz,na,nb]))



if __name__ == '__main__':
    import time
    from wannier_utils.wannier_system import WannierSystem

    if len(sys.argv) < 2:
        raise Exception("usage: Jij.py config_toml")

    with open(sys.argv[1], "rb") as fp:
        conf_toml = tomli.load(fp)

    start_time = time.time()

    prefix = conf_toml["prefix"]
    if conf_toml["dir"]:
        prefix = conf_toml["dir"] + "/" + prefix

    ws = WannierSystem(file_hr = prefix + "_hr.dat", file_nnkp = prefix + ".nnkp")

    jij = Jij(
                ws,
                kmesh = [conf_toml["nk1"], conf_toml["nk2"], conf_toml["nk3"]],
                mu = conf_toml["mu"],
                temperature_in_K = conf_toml["T"],
                nproc = conf_toml.get("nproc", 1),
             )
    jij.calc_all(rcut = conf_toml.get("rcut", 0))

    elapsed_time = time.time()
               
    print("")
    print("  elapsed_time:{:15.2f} [sec]".format(elapsed_time))

