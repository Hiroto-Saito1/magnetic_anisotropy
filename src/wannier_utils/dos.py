#!/usr/bin/env python

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf
from typing import Iterable, Optional

from wannier_utils.logger import get_logger
from wannier_utils.wannier_system import WannierSystem
from wannier_utils.wannier_kmesh import WannierKmesh

logger = get_logger(__name__)


class DOS(WannierKmesh):
    """
    DOS and related quantities calculation, based on the energy[nk, nx, ny, nz].

    Attributes:
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        kmesh: Iterable[int], 
        use_sym: bool = False, 
        magmoms: Optional[np.ndarray] = None, 
        is_projection_each: bool = False, 
        projection_matrices: Optional[np.ndarray] = None, 
        nproc: int = 1, 
    ):
        """
        Constructor.


        Args:
            ws (WannierSystem): a WannierSystem instance.
            kmesh (Iterable[int]): numpy array of grids (and shifts) along with reciprocal vectors.
            use_sym (bool): whether to reduce k points with spacegroup symmetry.
            magmoms (Optional[np.ndarray]): magnetic moments to determine magnetic spacegroup symmetry.
            is_projection_each (bool): _description_. defaults to False.
            projection_matrices (Optional[np.ndarray]): _description_. defaults to None.
            nproc (int): the number of process parallelization.
        """
        super().__init__(ws, kmesh, magmoms=magmoms, use_sym=use_sym, nproc=nproc)
        self.kmesh = kmesh
        self.ntetra = self.mp_points.nk*6
        self.convert_hamiltonian_to_energy_all(is_projection_each, projection_matrices)

    @property
    def etetra_all(self):
        if not hasattr(self, "_etetra_all"):
            self._etetra_all = self._convert_energy_to_etetra_all()
        return self._etetra_all

    @property
    def tetra_index(self):
        if not hasattr(self, "_tetra_index"):
            ind = lambda i, j, k : i*self.kmesh[1]*self.kmesh[2] + j*self.kmesh[2] + k
            ind_all = []
            for i in range(self.kmesh[0]):
                ip = (i + 1) % self.kmesh[0]
                for j in range(self.kmesh[1]):
                    jp = (j + 1) % self.kmesh[1]
                    for k in range(self.kmesh[2]):
                        kp = (k + 1) % self.kmesh[2]
                        ind_all.append([ind(i, j, k), ind(ip, j, k), ind(i, jp, k), ind(ip, j, kp)])
                        ind_all.append([ind(ip, j, k), ind(i, jp, k), ind(ip, jp, k), ind(ip, j, kp)])
                        ind_all.append([ind(i, j, k), ind(i, jp, k), ind(i, j, kp), ind(ip, j, kp)])
                        ind_all.append([ind(i, jp, k), ind(ip, jp, k), ind(ip, j, kp), ind(ip, jp, kp)])
                        ind_all.append([ind(i, jp, k), ind(ip, j, kp), ind(i, jp, kp), ind(ip, jp, kp)])
                        ind_all.append([ind(i, jp, k), ind(i, j, kp), ind(ip, j, kp), ind(i, jp, kp)])
            self._tetra_index = np.array(ind_all)
        return self._tetra_index

    def calc_dos(self, mu: float):
        dos = 0
        for nb in range(self.ws.num_wann):
            for ik in range(self.ntetra):
                dos += self._calc_dos_tetra(self.etetra_all[nb, ik, :], mu)
        return dos/self.ntetra

    def calc_dos_fast(self, mu, band_range = None):
        dos = 0
        if band_range == None:
            band_range = range(self.ws.num_wann)
        for nb in band_range:
            if (np.all(self.etetra_all[nb, :, :] < mu)):
                dos += 0.0
            elif (np.all(self.etetra_all[nb, :, :] > mu)):
                dos += 0.0
            else:
                for ik in range(self.ntetra):
                    dos += self._calc_dos_tetra(self.etetra_all[nb, ik, :], mu)
        return dos/self.ntetra

    def calc_dos_weight(self, mu: float, method: str = "tetra", delta: float = 0.1):
        return np.sum(self._calc_dos_weight(mu, method, delta))

    def calc_dos_projection(self, mu: float, method: str = "tetra", delta: float = 0.1):
        return np.einsum(
            "kl,kml->m", 
            self._calc_dos_weight(mu, method, delta), self.projection_weight, 
            optimize=True, 
        )

    def _calc_dos_weight(self, mu: float, method: str, delta: float):
        if method == "tetra":
            weight = self._calc_tetra_weight(mu, weight_method="dos")
        else:
            weight = self._calc_weight_smearing(
                mu, 
                delta=delta, 
                smearing=method, 
                weight_method="dos", 
            )/self.mp_points.nk
        return weight

    def calc_rho(self, mu: float):
        rho = 0
        for nb in range(self.ws.num_wann):
            for ik in range(self.ntetra):
                rho += self._calc_rho_tetra(self.etetra_all[nb, ik, :], mu)
        return rho/self.ntetra

    def calc_rho_fast(self, mu: float):
        rho = 0
        for nb in range(self.ws.num_wann):
            if (np.all(self.etetra_all[nb, :, :] < mu)):
                rho += self.ntetra
            elif (np.all(self.etetra_all[nb, :, :] > mu)):
                rho += 0.0
            else:
                for ik in range(self.ntetra):
                    rho += self._calc_rho_tetra(self.etetra_all[nb, ik, :], mu)
        return rho/self.ntetra

    def calc_rho_weight(self, mu: float, method: str = "tetra", delta: float = 0.1):
        return np.sum(self._calc_rho_weight(mu, method, delta))

    def calc_rho_projection(self, mu: float, method: str = "tetra", delta: float = 0.1):
        return np.einsum(
            "kl,kml->m", 
            self._calc_rho_weight(mu, method, delta), self.projection_weight, 
            optimize=True, 
        )

    def _calc_rho_weight(self, mu: float, method: str, delta: float):
        if method == "tetra":
            weight = self._calc_tetra_weight(mu, weight_method="sum")
        else:
            weight = self._calc_weight_smearing(
                mu, 
                delta=delta, 
                smearing=method, 
                weight_method="sum", 
            )/self.mp_points.nk
        return weight

    def calc_ef(
        self, 
        rho: float, 
        mu_lower: float = -100, 
        mu_upper: float = 100, 
        atol: float = 1e-8, 
    ):
        """
        Args:
            rho: electron density.
            mu_lower: _description_. Defaults to -100.
            mu_upper: _description_. Defaults to 100.
            atol: _description_. Defaults to 1e-8.

        Returns:
            float: Fermi level.
        """
        drho = lambda mu : self.calc_rho_fast(mu) - rho
        mu, result = brentq(drho, mu_lower, mu_upper, xtol=atol, full_output=True)
        if not result.converged:
            raise ValueError("mu is not converged.")
        return mu

    def calc_gap(self, ne: float):
        ev = np.max(self.energy_all[:, int(ne) - 1])
        ec = np.min(self.energy_all[:, int(ne)])
        logger.debug("Energy of VBT =", ev)
        logger.debug("Energy of CBM =", ec)
        return max(0, ec - ev)

    def convert_hamiltonian_to_energy_all(
        self, 
        is_projection_each: bool = False, 
        projection_matrices: Optional[np.ndarray] = None, 
    ):
        if is_projection_each and (projection_matrices is not None):
            raise ValueError("Both is_projection_each and projection_matrices are defined.")
        e_all = []
        proj_weight = []
        for k in self.mp_points.full_kvec:
            ham_k = self.ws.calc_ham_k(k, diagonalize=True)
            e_all.append(ham_k.ek[:])

            # weight[ns,nb] = sum_i v*[i,nb] p[i] v[i,nb]  with p[i] = delta_{i,n}
            if is_projection_each:
                proj_weight.append((ham_k.uk.conj()*ham_k.uk).real)
            elif projection_matrices is not None:
                proj_weight.append(np.einsum(
                    "in,aij,jn->an", 
                    ham_k.uk.conj(), projection_matrices, ham_k.uk, 
                    optimize=True
                ).real)
        self.energy_all = np.array(e_all)
        self.projection_weight = np.array(proj_weight)

    def _calc_dos_tetra(self, et: np.ndarray, mu: float):
        e21 = et[1] - et[0]
        e31 = et[2] - et[0]
        e32 = et[2] - et[1]
        e41 = et[3] - et[0]
        e42 = et[3] - et[1]
        e43 = et[3] - et[2]
        dos = 0
        if (et[2] <= mu and mu < et[3]):
            # dos = 3*(et[3]-mu)**2/(et[3] - et[0])/(et[3] - et[1])/(et[3] - et[2])
            dos = 3*(et[3] - mu)**2/e41/e42/e43
        elif (et[1] <= mu and mu < et[2]):
            # dos = {3*(et[1] - et[0]) + 6*(mu - et[1]) - 3*(et[2] - et[0] + et[3] - et[1])\
            #       /(et[2] - et[1])/(et[3] - et[1])*(mu - et[1])**2}\
            #       /(et[2] - et[0])/(et[3] - et[0])
            dos = (3*e21 + 6*(mu - et[1]) - 3*(e31 + e42)/e32/e42*(mu - et[1])* 2)/e31/e41
        elif (et[0] <= mu and mu < et[1]):
            # dos = 3*(mu - et[0])**2/(et[1] - et[0])/(et[2] - et[0])/(et[3] - et[0])
            dos = 3*(mu - et[0])**2/e21/e31/e41
        else:
            dos = 0
        return dos

    def _calc_rho_tetra(self, et: np.ndarray, mu: float):
        e21 = et[1] - et[0]
        e31 = et[2] - et[0]
        e32 = et[2] - et[1]
        e41 = et[3] - et[0]
        e42 = et[3] - et[1]
        e43 = et[3] - et[2]
        if (et[3] <= mu):
            rho = 1.0
        elif (et[2] <= mu and mu < et[3]):
            rho = (1 - (et[3] - mu)**3/e41/e42/e43)
        elif (et[1] <= mu and mu < et[2]):
            rho = (
                e21**2 \
                + 3*e21*(mu - et[1]) \
                + 3*(mu - et[1])**2 \
                - (e31 + e42)/e32/e42*(mu - et[1])**3
            )/e31/e41
        elif (et[0] <= mu and mu < et[1]):
            rho = (mu - et[0])**3/e21/e31/e41
        else:
            rho = 0
        return rho

    def _add_weight_sum(
        self, 
        ind: Iterable[int], 
        mu: float, 
        energy_all: np.ndarray, 
        weight_all: np.ndarray, 
    ):
        """
        Args:
            ind: _description_.
            mu: Fermi level.
            energy_all: _description_.
            weight_all: weight for \int_{-inf}^{mu}de.
        """
        energy = np.array(
            [energy_all[ind[0]], energy_all[ind[1]], energy_all[ind[2]], energy_all[ind[3]]]
        )
        sort_ind = np.argsort(energy)
        et = energy[sort_ind]
        e21 = et[1] - et[0]
        e31 = et[2] - et[0]
        e32 = et[2] - et[1]
        e41 = et[3] - et[0]
        e42 = et[3] - et[1]
        e43 = et[3] - et[2]
        e1 = et[0]
        e2 = et[1]
        e3 = et[2]
        e4 = et[3]
        if (e4 <= mu):
            for i in range(4):
                weight_all[ind[i]] += 1
        if (e3 <= mu and mu < e4):
            c = (e4 - mu)**3/e41/e42/e43
            weight_all[ind[0]] += 1 - c*(e4 - mu)/e41
            weight_all[ind[1]] += 1 - c*(e4 - mu)/e42
            weight_all[ind[2]] += 1 - c*(e4 - mu)/e43
            weight_all[ind[3]] += 1 - c*(4 - (1/e41 + 1/e42 + 1/e43)*(e4 - mu))
        if (e2 <= mu and mu < e3):
            c1 = (mu - e1)**2/e41/e31
            c2 = (mu - e1)*(mu - e2)*(e3 - mu)/e41/e32/e31
            c3 = (mu - e2)**2*(e4 - mu)/e42/e32/e41
            weight_all[ind[0]] += c1 + (c1 + c2)*(e3 - mu)/e31 + (c1 + c2 + c3)*(e4 - mu)/e41
            weight_all[ind[1]] += c1 + c2 + c3 + (c2 + c3)*(e3 - mu)/e32 + c3*(e4 - mu)/e42
            weight_all[ind[2]] += (c1 + c2)*(mu - e1)/e31 + (c2 + c3)*(mu - e2)/e32
            weight_all[ind[3]] += (c1 + c2 + c3)*(mu - e1)/e41 + c3*(mu - e2)/e42
        if (e1 <= mu and mu < e2):
            c = (mu - e1)**3/e21/e31/e41
            weight_all[ind[0]] += c*(4 - (mu - e1)*(1/e21 + 1/e31 + 1/e41))
            weight_all[ind[1]] += c*(mu - e1)/e21
            weight_all[ind[2]] += c*(mu - e1)/e31
            weight_all[ind[3]] += c*(mu - e1)/e41
        if (mu <= e1):
            pass

    def _add_weight_dos(
        self, 
        ind: Iterable[int], 
        mu: float, 
        energy_all: np.ndarray, 
        weight_all: np.ndarray, 
    ):
        """
        Args:
            ind: _description_.
            mu: Fermi level.
            energy_all: _description_.
            weight_all: weight for \int_{-inf}^{mu}de delta(e - mu)*(d weight_sum_all/d mu)
        """
        energy = np.array([energy_all[ind[0]], energy_all[ind[1]], energy_all[ind[2]], energy_all[ind[3]]])
        sort_ind = np.argsort(energy)
        et = energy[sort_ind]
        e21 = et[1] - et[0]
        e31 = et[2] - et[0]
        e32 = et[2] - et[1]
        e41 = et[3] - et[0]
        e42 = et[3] - et[1]
        e43 = et[3] - et[2]
        e1 = et[0]
        e2 = et[1]
        e3 = et[2]
        e4 = et[3]
        if (e4 <= mu):
            pass
        if (e3 <= mu and mu < e4):
            c = (e4 - mu)**3/e41/e42/e43
            weight_all[ind[0]] += 4*c/e41
            weight_all[ind[1]] += 4*c/e42
            weight_all[ind[2]] += 4*c/e43
            weight_all[ind[3]] += 12*(e4 - mu)**2/e41/e42/e43 - 4*c*(1/e41 + 1/e42 + 1/e43)
        if (e2 <= mu and mu < e3):
            c1 = (mu - e1)**2/e41/e31
            c2 = (mu - e1)*(mu - e2)*(e3 - mu)/e41/e32/e31
            c3 = (mu - e2)**2*(e4 - mu)/e42/e32/e41
            dc1 = 2*(mu - e1)/e41/e31
            dc2 = ((mu - e2)*(e3 - mu) + (mu - e1)*(e3 - mu) - (mu - e1)*(mu - e2))/e41/e32/e31
            dc3 = (2*(mu - e2)*(e4 - mu) - (mu - e2)**2)/e42/e32/e41
            weight_all[ind[0]] += dc1 \
                                  + (dc1 + dc2)*(e3 - mu)/e31 \
                                  + (dc1 + dc2 + dc3)*(e4 - mu)/e41 \
                                  - (c1 + c2)/e31 \
                                  - (c1 + c2 + c3)/e41
            weight_all[ind[1]] += dc1 + dc2 + dc3 \
                                  + (dc2 + dc3)*(e3 - mu)/e32 \
                                  + dc3*(e4 - mu)/e42 \
                                  - (c2 + c3)/e32 \
                                  - c3/e42
            weight_all[ind[2]] += (dc1 + dc2)*(mu - e1)/e31 \
                                  + (dc2 + dc3)*(mu - e2)/e32 \
                                  + (c1 + c2)/e31 \
                                  + (c2 + c3)/e32
            weight_all[ind[3]] += (dc1 + dc2 + dc3)*(mu - e1)/e41 \
                                  + dc3*(mu - e2)/e42 \
                                  + (c1 + c2 + c3)/e41 \
                                  + c3/e42
        if (e1 <= mu and mu < e2):
            c = (mu - e1)**3/e21/e31/e41
            weight_all[ind[0]] += 12*(e1 - mu)**2/e21/e31/e41 - 4*c*(1/e21 + 1/e31 + 1/e41)
            weight_all[ind[1]] += 4*c/e21
            weight_all[ind[2]] += 4*c/e31
            weight_all[ind[3]] += 4*c/e41
        if (mu <= e1):
            pass

    def _calc_tetra_weight(self, mu: float, weight_method: str):
        """
        Args:
            mu: upper limit of tetrahedron sum.
            weight_method: _description_.

        Returns:
            weight_all:
        """
        weight_all = np.zeros((self.mp_points.nk, self.ws.num_wann), dtype=np.float64)
        if (weight_method == "sum"):
            weight_func = self._add_weight_sum
        elif (weight_method == "dos"):
            weight_func = self._add_weight_dos
        else:
            raise ValueError("Method {} is not defined.".format(weight_method))

        for nb in range(self.ws.num_wann):
            if np.all(self.energy_all[:, nb] < mu):
                if (weight_method == "sum"):
                    weight_all[:, nb] = 4*6
                continue
            if np.all(self.energy_all[:, nb] > mu):
                continue
            for ind in self.tetra_index:
                ind_nb = [(ind[0], nb), (ind[1], nb), (ind[2], nb), (ind[3], nb)]
                weight_func(ind_nb, mu, self.energy_all, weight_all)

        weight_all = weight_all/(self.ntetra*4)
        return weight_all

    def _convert_energy_to_etetra_all(self):
        etetra_all = np.empty((self.ws.num_wann, self.ntetra, 4), dtype=np.float64)
        for i in range(self.ws.num_wann):
            etetra_all[i, :] = self._convert_energy_to_etetra(self.energy_all[:, i])
        return etetra_all

    def _convert_energy_to_etetra(self, energy):
        etetra = np.empty((self.ntetra, 4), dtype=np.float64)
        for n, ind in enumerate(self.tetra_index):
            etetra[n,:] = np.sort(
                [energy[ind[0]], energy[ind[1]], energy[ind[2]], energy[ind[3]]]
            )
        return etetra

    def _calc_weight_smearing(
        self, 
        ef: float, 
        delta: float, 
        smearing: float, 
        weight_method: str = "dos", 
    ):
        x = (self.energy_all - ef)/delta
        if weight_method == "dos":
            if smearing == "smearing" or smearing == "gaussian":
                return np.exp(-x**2)/(np.sqrt(np.pi)*delta)
            elif smearing == "fermi-dirac":
                return 1/(np.cosh(x) + 1)/(2*delta)
            raise ValueError("Method {} is not supported.".format(smearing))
        elif weight_method == "sum":
            if smearing == "smearing" or smearing == "gaussian":
                return (1 + erf(-x/np.sqrt(2)))/2
            elif smearing == "fermi-dirac":
                return (1 - np.tanh(x/2))/2
            raise ValueError("weight_method {} is not supported.".format(weight_method))
