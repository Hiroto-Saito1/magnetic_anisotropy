#!/usr/bin/env python

from typing import Optional

import numpy as np
from scipy.constants import Boltzmann, eV
from sparse_ir import FiniteTempBasis, MatsubaraSampling, TauSampling

Kelvin2eV = Boltzmann/eV


class Temperature:
    """
    IR basis settings.

    Attributes:
        temperature_eV: temperature in unit of eV.
        basis: IR basis instance.
        sample_matsu: sampling Matsubara indices.
        omegas_sp: sampling Matsubara frequencies.
        sample_tau: sampling imaginary times.
    """
    def __init__(self, statistics: str, temperature_K: float, wmax: float, eps: float = 1e-10):
        """
        Constructor.

        Args:
            statisics (str): whether "F" (Fermion) or "B" (Boson).
            temperature_K (float): in unit of Kelvin.
            wmax (float): energy range in unit of eV.
            eps (float): precision of Green's function in IR basis representation. defaults to 1e-10.
        """
        self.temperature_eV = temperature_K*Kelvin2eV
        beta = 1/self.temperature_eV

        self.basis = FiniteTempBasis(statistics, beta, wmax, eps=eps)
        self.sample_matsu = MatsubaraSampling(self.basis)
        self.omegas_sp = self.sample_matsu.sampling_points*np.pi/beta
        self.sample_tau = TauSampling(self.basis)

    def get_green_w(self, hk: np.ndarray, chem_pot: float, pot_w: Optional[np.ndarray] = None) \
    -> np.ndarray:
        """
        Return Green's function for given reciprocal-space Hamiltonian H(k).

        Args:
            hk (np.ndarray): Hamiltonian at a k point.
            chem_pot (float): chemical potential in unit of eV.
            pot_w (Optional[np.ndarray]): potential depending Matsubara frequencies (for CPA).

        Returns:
            np.ndarray: Green's function G(k, iw) in (nw, norb, norb) shape.
        """
        identity = np.identity(hk.shape[0], dtype=np.complex128)
        inv_greens_w = (1j*self.omegas_sp[:, None, None] + chem_pot)*identity[None, :, :] - hk[None, :, :]
        if pot_w: inv_greens_w += pot_w
        return np.linalg.inv(inv_greens_w)
