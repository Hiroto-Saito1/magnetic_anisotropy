#!/usr/bin/env python

import numpy as np
import irbasis
import scipy.constants

Kelvin2eV = scipy.constants.Boltzmann / scipy.constants.eV

class Temperature(object):
    """
    temperature and irbasis setting
    """
    def __init__(self, temperature, IR_Lambda=1e+5, IR_eps=1e-8, use_fermionic_basis=True):
        """
        input:
            temperature: in Kelvin
        """
        self.temperature_in_eV = temperature * Kelvin2eV
        self.beta = 1/self.temperature_in_eV

        IR_wmax = IR_Lambda * self.temperature_in_eV
        self.basis_F = irbasis.load('F', IR_Lambda)
        print("# for IRbasis3")
        print("# T = {}".format(temperature))
        print("# Lambda = {}".format(IR_Lambda))
        print("# wmax = {}".format(IR_wmax))
        print("# irFdim = {}".format(self.basis_F.dim()))

        # calculate sampling matsubara points
        self.smpl_F = self.basis_F.sampling_points_matsubara(whichl=self.basis_F.dim()-1)
        self.omegaF_sp = (2*self.smpl_F+1) * np.pi / self.beta

        self.hatF = np.sqrt(self.beta) * self.basis_F.compute_unl(self.smpl_F)
        self.inv_hatF = np.linalg.pinv(self.hatF)

        #ultau0 = np.sqrt(2*self.temperature) * np.array([self.irF.ulx(l, -1) for l in range(self.irFdim)])  # U_l(tau=0)
        #ultau1 = np.sqrt(2*self.temperature) * np.array([self.irF.ulx(l, 1) for l in range(self.irFdim)])  # U_l(tau=0)
        ultau0 = self.Ultau(0)
        ultau1 = self.Ultau(self.beta)
        lfactor = (ultau0 - ultau1)/2

        self.sum_factor = np.einsum("l,ln->n", lfactor, self.inv_hatF, optimize='optimal')
        self.sum_factor2 = np.einsum("l,ln->n", ultau0, self.inv_hatF, optimize='optimal')

    def iwn2l(self, g_iwn):
        return np.einsum("ln,...n->...l", self.inv_hatF, g_iwn)

    def Ultau(self, tau):
        x = 2*tau/self.beta - 1
        return np.sqrt(2/self.beta) * self.basis_F.ulx(l=None, x=x)
