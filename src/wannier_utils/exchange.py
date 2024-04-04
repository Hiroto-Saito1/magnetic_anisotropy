#!/usr/bin/env python

"""
Description:
    Compute Heisenberg exchange interaction by wannierized Lichtenstein's formula.

Usage:
    exchange_k.py <exc_in_toml> (--q0 | --rq=<qvec_npz> | --q=<qvec_npz>)
    exchange_k.py [-h | --help]

Options:
    --q0                Compute at q = 0.
    --rq=<qvec_npz>     Compute at given real lattice points.
    --q=<qvec_npz>      Compute at given reciprocal lattice points.
    -h --help           Show this help screen.

Author:
    Katsuhiro Arimoto
"""

from itertools import product, chain
from pathlib import Path
from re import search
from typing import List, Iterable, Optional, Tuple
from warnings import warn

import numpy as np
from pymatgen.core.lattice import Lattice
from tomli import load
from tomli_w import dump

from wannier_utils.fourier import Fourier
from wannier_utils.temperature_spir import Temperature
from wannier_utils.wannier_kmesh import WannierKmesh
from wannier_utils.wannier_system import WannierSystem
from mymodule import convert_numpy_object, get_my_logger

logger = get_my_logger(__name__)


def main():
    from time import perf_counter

    from docopt import docopt

    logger.info(f"{Path(__file__).name} started.")

    start_time = perf_counter()
    args = docopt(__doc__)
    exc_in_toml = Path(args["<exc_in_toml>"])
    with open(exc_in_toml, mode="rb") as fp:
        exc_in_dict = load(fp)
    work_dir = Path(exc_in_dict["work_dir"])
    w90_prefix = exc_in_dict["w90_prefix"]
    ws = WannierSystem(
        hr_dat=work_dir/f"{w90_prefix}_hr.dat", 
        nnkp_file=work_dir/f"{w90_prefix}.nnkp",
        win_file=work_dir/f"{w90_prefix}.win", 
    )
    exc = Exchange(
        ws, 
        exc_in_dict["kmesh"], 
        exc_in_dict["chem_pot"], 
        exc_in_dict["temperature_K"], 
        wmax=exc_in_dict["wmax"], 
        s_axis=exc_in_dict["s_axis"], 
        nproc=exc_in_dict["nproc"], 
    )
    exc_in_prefix = exc_in_toml.stem.replace("_in", "")

    if args["--q0"]:
        g_up_r0w, g_dn_r0w, exc_ij, exc_i, exc_r0ij = exc.calc_exchange_q0()
        np.savez_compressed(exc_in_toml.parent/f"{exc_in_prefix}_q0", exc_ij=exc_ij, exc_i=exc_i)
        exc_out_id = "q0"
    if args["--rq"]:
        if exc.mp_points.nk < np.prod(exc.ws.win.mp_grid):
            warn("k mesh should be fine than MP grid of Wannierization.")
        if exc_in_dict["sc_grid"] != exc_in_dict["kmesh"]:
            warn("WS Supercell should agree with k mesh for exact Matsubara-Green's function.")
        mp_grid = np.fmax(exc_in_dict["sc_grid"], exc.ws.win.mp_grid)
        irvec, ir0, ndegen = get_irvec(mp_grid, exc.ws.a)
        g_up_r0w, g_dn_r0w, exc_rij, exc_i = exc.calc_exchange_r(irvec, ir0, ndegen)
        exc_r0ij = exc_rij[ir0]
        np.savez_compressed(
            exc_in_toml.parent/f"{exc_in_prefix}_r", 
            irvec=irvec, 
            ir0=ir0, 
            ndegen=ndegen, 
            exc_ij=exc_rij, 
            exc_i=exc_i, 
        )
        qvec_npz = Path(args["--rq"])
        qvec = np.load(qvec_npz)["qvec"]
        exc_qij = convert_exchange_r_to_q(irvec, ndegen, exc_rij, qvec)
        exc_qij[:, range(exc.ws.natom), range(exc.ws.natom)] \
            -= exc_r0ij[None, range(exc.ws.natom), range(exc.ws.natom)]
        exc_out_id = "rq_{}".format(search("[1-3]d", qvec_npz.stem).group())
        np.savez_compressed(
            exc_in_toml.parent/f"{exc_in_prefix}_{exc_out_id}", 
            exc_ij=exc_qij, 
        )
    if args["--q"]:
        qvec_npz = Path(args["--q"])
        qvec = np.load(qvec_npz)["qvec"]
        g_up_r0w, g_dn_r0w, exc_ij, exc_i, exc_r0ij = exc.calc_exchange_q(qvec)
        exc_out_id = "q_{}".format(search("[1-3]d", qvec_npz.stem).group())
        np.savez_compressed(
            exc_in_toml.parent/f"{exc_in_prefix}_{exc_out_id}", 
            exc_ij=exc_ij, 
            exc_i=exc_i, 
        )

    ne, ns = exc.calc_ne_ns(g_up_r0w, g_dn_r0w)
    onsite_mag_splits = exc.get_onsite_mag_splits()
    exc_out_dict = {
        "wmax_eV": convert_numpy_object(exc.tmpr.basis.wmax), 
        "num_electron": convert_numpy_object(ne), 
        "num_spin": convert_numpy_object(ns), 
        "onsite_mag_splits": list(convert_numpy_object(onsite_mag_splits)), 
        "exc_r0ii": list(convert_numpy_object(np.abs(exc_r0ij.diagonal()))), 
    }
    exc_out_toml = exc_in_toml.parent/exc_in_toml.name.replace("in", "out_" + exc_out_id)
    with open(exc_out_toml, mode="wb") as fp:
        dump(exc_out_dict, fp)

    end_time = perf_counter()
    logger.info(f"elapsed time: {end_time - start_time} [sec].")
    logger.info(f"{Path(__file__).name} ended.")


class Exchange(WannierKmesh):
    """
    Evaluate Heisenberg exchange interaction with Lichtenstein's formula and below settings.
    1. DFT calculation is done in colinear magnetization along z axis, no SOC.
    2. Wannier90 calculation is one shot.
    3. Local force approximation for spin-dependent hopping is applied.

    Attributes:

    Note:
        Definition of exchange interaction is below.
            E := -\sum_{R, R'}\sum_{i, j}J_{ij}(R)e_{i}(R)e_{j}(R + R')
            J_{i}(R) := \sum_{j \neq i}(J_{ij}(R) + J_{ji}(R))
            J_{ij}(q) := \sum_{R}J_{ij}(R)e^{-iqR}
            J_{i}(q) := \sum_{R}J_{i}(R)e^{-iqR}
            T_{c}(q) = 2/3*1/4*<J_{i}(q)>/k_{B}
        In the last equation, 1/4 is a adjustment factor from conventional Heisenberg model.
            E := -2\sum_{R, R'}\sum_{i, j}J_{ij}(R)e_{i}(R)e_{j}(R + R')
            J_{i}(R) := \sum_{j \neq i}(J_{ij}(R))
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        kmesh: Iterable[int], 
        chem_pot: float, 
        temperature_K: float, 
        wmax: float = 1000, 
        s_axis: np.ndarray = np.array([0, 0, 1], dtype=np.float64), 
        nproc: int = 1, 
    ):
        """
        Constructor.

        Args:
            ws (WannierSystem): a WannierSystem instance.
            kmesh (Iterable[int]): list of k mesh (and shifts) in fractional coordinates.
            chem_pot (float): chemical potential in unit of eV.
            temperature_K (float): temperature in unit of Kelvin.
            wmax (float): maximum of energy range treated in IR basis. defaults to 1000.
            s_axis (np.ndarray): direction of spin quantization axis. defaults to z axis; [0, 0, 1].
            nproc (int): the number of parallelization. defaults to 1.
        """
        if ws.win.num_iter:
            raise ValueError("Only for one shot wannierization.")
        if ws.num_wann%2:
            raise ValueError("The number of Wannier orbital is odd.")

        super().__init__(ws, kmesh, nproc=nproc)
        self.chem_pot = chem_pot
        self.tmpr = Temperature("F", temperature_K, wmax)
        self.up_proj, self.dn_proj = self._get_projectors(s_axis)
        self.orb_idcs = self._get_orb_idcs()
        self.mag_pots_rs = self._calc_mag_pots_rs()

    def _get_projectors(self, s_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return matrix projecting Hamiltonian to a spin axis.

        Args:
            s_axis (np.ndarray): spin axis direction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: projection operator for wannier-based matrix.
        """
        sx_up = np.array([1, 1], dtype=np.complex128)/np.sqrt(2)
        sx_dn = np.array([1, -1], dtype=np.complex128)/np.sqrt(2)
        sy_up = np.array([1, 1j], dtype=np.complex128)/np.sqrt(2)
        sy_dn = np.array([1, -1j], dtype=np.complex128)/np.sqrt(2)
        sz_up = np.array([1, 0], dtype=np.complex128)
        sz_dn = np.array([0, 1], dtype=np.complex128)
        s_ups = np.array([sx_up, sy_up, sz_up], dtype=np.complex128)
        s_dns = np.array([sx_dn, sy_dn, sz_dn], dtype=np.complex128)

        up_state = np.einsum("d,ds->s", s_axis, s_ups, optimize=True)
        dn_state = np.einsum("d,ds->s", s_axis, s_dns, optimize=True)
        up_proj = np.kron(np.identity(self.ws.num_wann//2), up_state)
        dn_proj = np.kron(np.identity(self.ws.num_wann//2), dn_state)
        return up_proj, dn_proj

    def _get_orb_idcs(self) -> List[List[int]]:
        """
        Return indices of orbitals included in each atom for projected Hamiltonian.

        Returns:
            List[List[int]]: indices of orbitals included in each atom.
        """
        orb_idcs = []
        for p in self.ws.nnkp.atom_pos:
            orb_idx = np.array(list(
                chain.from_iterable([self.ws.nnkp.atom_orb[o] for o in p])
            ))[::2]//2
            orb_idcs.append(list(orb_idx))
        return orb_idcs

    def _calc_mag_pots_rs(self) -> List[np.ndarray]:
        """
        Return magnetic potentials on atoms.

        Returns:
            List[np.ndarray]: magnetic potentials of atoms in (nr, norb, norb) shape.
        """
        hrs_up = np.einsum(
            "sp,rpq,tq->rst", 
            self.up_proj, self.ws.ham_r.hrs, self.up_proj, 
            optimize=True, 
        )
        hrs_dn = np.einsum(
            "sp,rpq,tq->rst", 
            self.dn_proj, self.ws.ham_r.hrs, self.dn_proj, 
            optimize=True, 
        )
        mag_pots_all = (hrs_up - hrs_dn)/2
        mag_pots_rs = []
        for orb_idx in self.orb_idcs:
            mag_pot_rs = mag_pots_all.take(orb_idx, axis=1).take(orb_idx, axis=2)
            mag_pots_rs.append(mag_pot_rs)
        return mag_pots_rs

    def _calc_k(self, k: np.ndarray, q: Optional[np.ndarray] = None) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate J_{ij}(k, q) and J_{i}(k, q).

        Args:
            k (np.ndarray): a k point.
            q (Optional[np.ndarray]): a q point. defaults to None (corresponding to q = 0).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                Matsubara-Green's functions at given k point, J_{ij}(k, q) and J_{i}(k, q).
        """
        g_up_w, g_dn_w = self._calc_greens_w(k)
        mag_pots_k = self._calc_mag_pots_k(k)

        if q is None:
            exc_ij = -(
                self._calc_trace_gsgs(g_up_w, g_dn_w) + self._calc_trace_gsgs(g_dn_w, g_up_w)
            )/2
            exc_i = self._calc_trace_gs(g_up_w, g_dn_w, mag_pots_k) - 2*np.diag(exc_ij)
            return g_up_w, g_dn_w, exc_ij, exc_i

        kq = get_kpt_1bz(k + q)
        g_up_q_w, g_dn_q_w = self._calc_greens_w(kq)
        exc_ij = -(
            self._calc_trace_gsgs(g_up_w, g_dn_q_w) + self._calc_trace_gsgs(g_dn_w, g_up_q_w)
        )/2
        exc_i = self._calc_trace_gs(g_up_q_w, g_dn_q_w, mag_pots_k) - 2*np.diag(exc_ij)
        return g_up_w, g_dn_w, exc_ij, exc_i

    def _calc_greens_w(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Matsubara-Green's functions for up and down spins at given k point.

        Args:
            k (np.ndarray): a k point.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Matsubara-Green's functions for up and down spins.
        """
        ham_k = self.ws.calc_ham_k(k)
        hk_up = np.einsum("sp,pq,tq->st", self.up_proj, ham_k.hk, self.up_proj, optimize=True)
        hk_dn = np.einsum("sp,pq,tq->st", self.dn_proj, ham_k.hk, self.dn_proj, optimize=True)
        g_up_w = self.tmpr.get_green_w(hk_up, self.chem_pot)
        g_dn_w = self.tmpr.get_green_w(hk_dn, self.chem_pot)
        return g_up_w, g_dn_w

    def _calc_mag_pots_k(self, k: np.ndarray) -> List[np.ndarray]:
        """
        Calculate Fourier transform of the magnetic potential.

        Args:
            k (np.ndarray): a k point.

        Returns:
            List[np.ndarray]: magnetic potentials of atoms at given k point.
        """
        fourier = Fourier(k, self.ws.ham_r.irvec, ndegen=self.ws.ham_r.ndegen)
        if not fourier.factor.ndim == 2:
            raise ValueError("Fourier factor has a wrong shape.")
        mag_pots_k = []
        for mag_pot_rs in self.mag_pots_rs:
            mag_pot_k = np.einsum("kr,rpq->pq", fourier.factor, mag_pot_rs, optimize=True)
            mag_pots_k.append(mag_pot_k)
        return mag_pots_k

    def _calc_trace_gsgs(self, g1_w: np.ndarray, g2_w: np.ndarray) -> np.ndarray:
        """
        Calculate trace and Matsunara sum in the formula of J_{ij}.

        Args:
            g1_w (np.ndarray): first Matsubara-Green's function in (..., nw, norb, norb) shape.
            g2_w (np.ndarray): second Matsubara-Green's function in (..., nw, norb, norb) shape.

        Returns:
            np.ndarray: trace in the formula of J_{ij}.
        """
        trace_ij = np.empty((self.ws.natom, self.ws.natom), dtype=np.complex128)
        for i, j in product(range(self.ws.natom), repeat=2):
            g1_w_ji = g1_w.take(self.orb_idcs[j], axis=-2).take(self.orb_idcs[i], axis=-1)
            g2_w_ij = g2_w.take(self.orb_idcs[i], axis=-2).take(self.orb_idcs[j], axis=-1)
            trace_w = np.einsum(
                "wpq,qs,wst,tp->w", 
                g1_w_ji, self.mag_pots_rs[i][self.ws.ham_r.ir0], g2_w_ij, self.mag_pots_rs[j][self.ws.ham_r.ir0], 
                optimize=True, 
            )
            trace_l = self.tmpr.sample_matsu.fit(trace_w, axis=0)
            trace_ij[i, j] = np.dot(trace_l, -self.tmpr.basis.u(self.tmpr.basis.beta))
        return trace_ij

    def _calc_trace_gs(self, g_up_w: np.ndarray, g_dn_w: np.ndarray, mag_pots: np.ndarray) \
    -> np.ndarray:
        """
        Calculate trace and Matsubara sum in the formula of J_{i}.

        Args:
            g_up_w: Matsubara-Green's function for up spins in (..., nw, norb, norb) shape.
            g_dn_w: Matsubara-Green's function for down spins in (..., nw, norb, norb) shape.

        Returns:
            np.ndarray: trace in the formula of J_{i}.
        """
        trace_i = np.empty(self.ws.natom, dtype=np.complex128)
        for i in range(self.ws.natom):
            g_up_w_ii = g_up_w.take(self.orb_idcs[i], axis=-2).take(self.orb_idcs[i], axis=-1)
            g_dn_w_ii = g_dn_w.take(self.orb_idcs[i], axis=-2).take(self.orb_idcs[i], axis=-1)
            trace_w = np.einsum(
                "wpq,qp->w", 
                -(g_up_w_ii - g_dn_w_ii), mag_pots[i], 
                optimize=True, 
            )
            trace_l = self.tmpr.sample_matsu.fit(trace_w, axis=0)
            trace_i[i] = np.dot(trace_l, -self.tmpr.basis.u(self.tmpr.basis.beta))
        return trace_i

    def get_onsite_mag_splits(self) -> np.ndarray:
        """
        Return energy splitting of onsite magnetic potentsials.

        Returns:
            np.ndarray: energy splitting of onsite magnetic potentials in unit of eV.
        """
        mag_splits = []
        for i, mag_pot_rs in enumerate(self.mag_pots_rs):
            mag_pot_r0_diag = mag_pot_rs[self.ws.ham_r.ir0].diagonal()
            assert np.allclose(
                mag_pot_r0_diag.imag, 
                np.zeros_like(mag_pot_r0_diag, dtype=np.float64), 
                atol=1e-6, 
            ), f"Magnetic potential of atom {i} has large imaginary part at R = 0."
            mag_splits.append(np.abs(mag_pot_r0_diag.real).max())
        return np.array(mag_splits, dtype=np.float64)

    def calc_ne_ns(self, g_up_r0w: np.ndarray, g_dn_r0w: np.ndarray) -> Tuple[float, float]:
        """
        Calculate total number of electrons and spins per unit cell.

        Args:
            g_up_r0w (np.ndarray): Matsubara-Green's function for up spins at R = 0.
            g_dn_r0w (np.ndarray): Matsubara-Green's function for down spins at R = 0.

        Return:
            Tuple[float, float]: the total number of electrons and spins.
        """
        tr_g_up_w = np.einsum("wpp->w", g_up_r0w, optimize=True)
        tr_g_dn_w = np.einsum("wpp->w", g_dn_r0w, optimize=True)
        tr_g_add_l = self.tmpr.sample_matsu.fit(tr_g_up_w + tr_g_dn_w)
        ne = np.dot(tr_g_add_l, -self.tmpr.basis.u(self.tmpr.basis.beta))
        assert ne.imag < 1e-5, "Large imaginary part in number of electrons."
        tr_g_diff_l = self.tmpr.sample_matsu.fit(tr_g_up_w - tr_g_dn_w)
        ns = np.dot(tr_g_diff_l, -self.tmpr.basis.u(self.tmpr.basis.beta))
        assert ns.imag < 1e-5, "Large imaginary part in number of spins."
        logger.info("Calculation of the numbers of electrons and spins is Done.")
        return ne.real, ns.real

    def calc_exchange_q0(self) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate J_{ij} and J_{i} at q = 0 (Gamma point).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                Matsubara-Green's functions at R = 0, J_{ij}(q = 0), J_{i}(q = 0), and J_{ij}(R = 0).
        """
        result = tuple(chain.from_iterable(self.parallel_k.apply_func_k(self._calc_k, args=(None,))))
        exc_q0ij = np.array(result[2::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk
        exc_q0i = np.array(result[3::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk

        # Eliminate self interaction term (c.f. Nomoto et al., PRL 125, 117204 (2020))
        g_up_r0w = np.array(result[0::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk
        g_dn_r0w = np.array(result[1::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk
        exc_r0ij = -(
            self._calc_trace_gsgs(g_up_r0w, g_dn_r0w) + self._calc_trace_gsgs(g_dn_r0w, g_up_r0w)
        )/2
        assert exc_r0ij.imag.max() < 1e-5, "Large imaginary part in exc_r0ij."
        exc_q0ij[range(self.ws.natom), range(self.ws.natom)] \
            -= exc_r0ij[range(self.ws.natom), range(self.ws.natom)]
        assert exc_q0ij.imag.max() < 1e-5, "Large imaginary part in exc_q0ij."
        assert exc_q0i.imag.max() < 1e-5, "Large imaginary part in exc_q0i."
        logger.info("Calculation of exchanges at q = 0 is Done.")

        return g_up_r0w, g_dn_r0w, exc_q0ij.real, exc_q0i.real, exc_r0ij.real

    def calc_exchange_r(
        self, 
        irvec: np.ndarray, 
        ir0: int, 
        ndegen: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate J_{ij}(R) and J_{i}(R = 0) (R = 0 is due to local force approximation).

        Args:
            irvec (np.ndarray): real-space grid points.
            ir0 (int): index of origin of the real-space grid points.
            ndegen (np.ndarray): degeneracies of the grid points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                Matsubara-Green's functions at R = 0, J_{ij}(R) and J_{i}(R = 0).
        """
        result = np.array(self.parallel_k.apply_func_k(self._calc_greens_w), dtype=np.complex128)
        g_up_kw, g_dn_kw = result[:, 0], result[:, 1]
        g_up_prw, g_up_mrw = self._convert_green_k_to_r(irvec, ndegen, g_up_kw)
        g_dn_prw, g_dn_mrw = self._convert_green_k_to_r(irvec, ndegen, g_dn_kw)
        exc_rij = np.empty((len(irvec), self.ws.natom, self.ws.natom), dtype=np.complex128)
        for ir, (gup, gdp, gum, gdm) in enumerate(zip(g_up_prw, g_dn_prw, g_up_mrw, g_dn_mrw)):
            exc_rij[ir] = -(
                self._calc_trace_gsgs(gum, gdp) + self._calc_trace_gsgs(gdm, gup)
            )/2
        assert exc_rij.imag.max() < 1e-5, "Large imaginary part in exc_rij."
        logger.info("Calculation of exc_rij is Done.")

        exc_i = self._calc_trace_gs(
            g_up_prw[ir0], 
            g_dn_prw[ir0], 
            [mag_pot_rs[self.ws.ham_r.ir0] for mag_pot_rs in self.mag_pots_rs], 
        ) - 2*np.diag(exc_rij[ir0])
        assert exc_i.imag.max() < 1e-5, "Large imaginary part in exc_i."
        logger.info("Calculation of exc_i is Done.")

        return g_up_prw[ir0], g_dn_prw[ir0], exc_rij.real, exc_i.real

    def _convert_green_k_to_r(
        self, 
        irvec: np.ndarray, 
        ndegen: np.ndarray, 
        g_kw: np.ndarray, 
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate inverse Fourier transform (from k to r) of given Matsubara-Green's function.

        Args:
            irvec (np.ndarray): real-space grid points.
            ndegen (np.ndarray): degeneracies of the grid points.
            g_kw (np.ndarray): Matsubara-Green's function in (nk, nw, norb, norb).

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                Matsubara-Green's functions in real space (R and -R).
        """
        fourier = Fourier(self.mp_points.full_kvec, irvec, ndegen=ndegen)
        if not fourier.inv_factor.ndim == 2:
            raise ValueError("Inverse Fourier factor has a wrong shape.")
        g_prw = np.einsum("rk,kwpq->rwpq", fourier.inv_factor, g_kw, optimize=True)
        g_mrw = np.einsum("rk,kwpq->rwpq", fourier.inv_factor.conj(), g_kw, optimize=True)
        return g_prw, g_mrw

    def calc_exchange_q(self, qvec: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate J_{ij}(q) and J_{i}(q) at given q points.

        Args:
            qvec (np.ndarray): q points.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
                Matsubara-Green's functions at R = 0, J_{ij}(q), J_{i}(q), and J_{ij}(R = 0).
        """
        exc_qij = np.empty((len(qvec), self.ws.natom, self.ws.natom), dtype=np.complex128)
        exc_qi = np.empty((len(qvec), self.ws.natom), dtype=np.complex128)
        for iq, q in enumerate(qvec):
            result = tuple(chain.from_iterable(
                self.parallel_k.apply_func_k(self._calc_k, args=(q,))
            ))
            exc_ij = np.array(result[2::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk
            exc_i = np.array(result[3::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk

            # Eliminate self interaction term (c.f. Nomoto et al., PRL 125, 117204 (2020))
            if not iq:
                g_up_r0w = np.array(result[0::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk
                g_dn_r0w = np.array(result[1::4], dtype=np.complex128).sum(axis=0)/self.mp_points.nk
                exc_r0ij = -(
                    self._calc_trace_gsgs(g_up_r0w, g_dn_r0w) + self._calc_trace_gsgs(g_dn_r0w, g_up_r0w)
                )/2
                assert exc_r0ij.imag.max() < 1e-5, "Large imaginary part in exc_r0ij."

            exc_ij[range(self.ws.natom), range(self.ws.natom)] \
                -= exc_r0ij[range(self.ws.natom), range(self.ws.natom)]
            exc_qij[iq] = exc_ij; exc_qi[iq] = exc_i
            logger.info(f"Calculation of exchanges at {iq}/{len(qvec) - 1} is Done.")

        return g_up_r0w, g_dn_r0w, exc_qij, exc_qi, exc_r0ij.real


def get_irvec(mp_grid: Iterable[int], a: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Calculate Wigner-Seitz grid points.

    Args:
        mp_grid (Iterable[int]): Monkhorst-Pack grid mesh.
        a (np.ndarray): primitive lattice vectors.

    Returns:
        Tuple[np.ndarray, int, np.ndarray]: 
            Wigner-Seitz grid points, its index of R = 0, and its degeneracies.
    """
    logger.info(f"We take {mp_grid[0]}*{mp_grid[1]}*{mp_grid[2]} Wigner-Seitz grid mesh.")
    mp_list = np.array(list(product(
        range(-mp_grid[0], mp_grid[0] + 1), 
        range(-mp_grid[1], mp_grid[1] + 1), 
        range(-mp_grid[2], mp_grid[2] + 1), 
    )))
    ilist = np.array(list(product(range(-2, 3), repeat=3)))
    real_metric = np.einsum("ad,bd->ab", a, a, optimize=True)

    # get distances from Wannier orbitals on supercell (L1*L2*L3)
    # sizes: rlist[npos, L1*L2*L3, 3], dist[npos, L1*L2*L3]
    rlist = mp_list[:, None, :] - np.einsum(
        "na,a->na", 
        ilist, mp_grid, 
        optimize=True, 
    )[None, :, :]
    dists = np.einsum("ija,ab,ijb->ij", rlist, real_metric, rlist, optimize=True)

    # dists[:, 62] = dist[:, i1 == i2 == i3 == 0]
    dist0 = dists[:, 62]
    inds = np.where(dist0 - np.min(dists, axis=1) < 1e-5)
    ndegen_all = np.sum(np.abs(dists - dist0[:, None]) < 1e-5, axis=1)

    irvec = mp_list[inds]
    ir0 = np.where(~irvec.any(axis=1))[0][0]
    ndegen = ndegen_all[inds]

    # check sum rule
    assert np.sum(1/ndegen) - np.product(mp_grid) < 1e-8, "Error in finding Wigner-Seitz points."
    return irvec, ir0, ndegen


def convert_exchange_r_to_q(
    rvec: np.ndarray, 
    ndegen: np.ndarray, 
    exc_rij: np.ndarray, 
    qvec: np.ndarray, 
) -> np.ndarray:
    qvec = qvec[None, :] if qvec.ndim == 1 else qvec
    exp_kr = np.exp(-2*np.pi*1j*np.einsum("qd,rd->qr", qvec, rvec, optimize=True))/ndegen
    return np.einsum("rij,qr->qij", exc_rij, exp_kr, optimize=True)


def get_kpt_1bz(
    kpt: np.ndarray, 
    is_kshifts: Iterable[bool] = [False]*3, 
    is_cartesian: bool = False, 
    lattice: Optional[Lattice] = None, 
) -> np.ndarray:
    if is_cartesian:
        kpt = lattice.reciprocal_lattice.get_fractional_coords(kpt)
    intervals_1bz = np.array([[0, 1]]*3, dtype=np.float64) \
                    - np.array(is_kshifts, dtype=np.float64)[:3, None]/2
    kpts_1bz = np.empty(3, dtype=np.float64)
    for d, (k, interval) in enumerate(zip(kpt, intervals_1bz)):
        while not interval[0] <= k < interval[1]:
            k = k + 1 if k < 0 else k - 1
        kpts_1bz[d] = k
    return kpts_1bz


if __name__ == "__main__":
    main()
