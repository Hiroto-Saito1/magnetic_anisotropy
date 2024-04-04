# -*- coding: utf-8 -*-

"""Module to calculate the Hamiltonian depending on the direction of magnetization.

Examples:
    MagRotation(
        hr_dat=Path("../tests/mp-2260/pwscf_rel_sym/mae/wan/pwscf_py_hr.dat"),
        )
"""
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from scipy.linalg import expm
from datetime import datetime
from itertools import product

from wannier_utils.hamiltonian import HamR
from wannier_utils.wannier_system import WannierSystem


class MagRotation:
    """This class reconstructs the non-collinear real-space Hamiltonian obtained from wannier_utils/hamiltonian.py
    depending on the direction of magnetization by splitting it into hopping, magnetization, and SOC.

    Args:
        tb_dat (Path, optional): Path to tb.dat.
        hr_dat (Path, optional): Path to hr.dat. Either tb.dat or hr.dat is needed.
        nnkp_file (Path, optional): Path of wannier.nnkp. Default to None.
        win_file (Path, optional): Path of wannier.win. Default to None.
        use_convert_ham_r (bool, optional): Whether to use convert_ham_r. Default to False.
        theta (float, optional): 1st rotation angle [rad]. Default to 0.
        phi (float, optional): 2nd rotation angle [rad]. Default to 0.
        axis_vec0 (list, optional): 1st rotation axis. Default to e_y.
        axis_vec1 (list, optional): 2nd roitation axis. Default to e_x.
        lam_para (float, optional): Parameter of the magnitude of SOC. Default to 1.0.
        extract_only_x_component (bool, optional): Whether to extract the x component of magnetization. Default to True.
        write_rotated_hr (bool, optional): Whether to write Hamiltonian with rotated magnetization. Default to False.
        file_name (str, optional): Name of rotated_hr. Default to "pwscf_rot_hr.dat".

    Attributes:
        num_wann: Number of Wannier orbitals.
        nrpts: Number of unit cells.
        ndegen: Degeneracies of Wigner-Seitz grid points in (nrpts) array.
        irvec: Lattice vectors for unit cells in (nrpts, 3) array.
        ir0: Index of H(0). If not found, return -1.
        hrs: Hamiltonian after mag rotation in (nrpts, num_wann, num_wann) array.
        mag_orig: Magnetization obtained by time reversal operation in (nrpts, num_wann, num_wann) array.
        mag_extracted: x component of magnetization in (nrpts, num_wann, num_wann) array.
        soc: SOC in (nrpts, num_wann, num_wann) array.
        hopping: Hopping in (nrpts, num_wann, num_wann) array.

    Notes:
        The magnetic anisotropy energy is more accurate if only the x component of the magnetization is considered.
    """

    def __init__(
        self,
        tb_dat: Optional[Path] = None,
        hr_dat: Optional[Path] = None,
        nnkp_file: Optional[Path] = None,
        win_file: Optional[Path] = None,
        use_convert_ham_r: Optional[bool] = False,
        theta: Optional[float] = 0.0,
        phi: Optional[float] = 0.0,
        axis_vec0: Optional[list] = [0, 1, 0],
        axis_vec1: Optional[list] = [1, 0, 0],
        lam_para: Optional[float] = 1.0,
        extract_only_x_component: Optional[bool] = True,
        write_rotated_hr: Optional[bool] = False,
        file_name: Optional[str] = "pwscf_rot_hr.dat",
    ):
        """Constructor that reads Hamiltonian using HamR,
        divides it into hopping (self.hopping), magnetization (self.mag_extracted), and SOC (self.soc),
        rotates the magnetization, and stores the rotated Hamiltonian in self.hrs.
        """
        if use_convert_ham_r:
            ws = WannierSystem(tb_dat, hr_dat, nnkp_file, win_file)
            ws._convert_ham_r()
            ham_r = ws.ham_r
        else:
            ham_r = HamR(tb_dat, hr_dat)

        self.num_wann = ham_r.num_wann
        self.nrpts = ham_r.nrpts
        self.ndegen = ham_r.ndegen
        self.irvec = ham_r.irvec
        self.ir0 = ham_r.ir0

        self.mag_orig, self.mag_extracted, self.hopping, self.soc = self.split_ham(
            ham_r, extract_only_x_component
        )
        self.hrs = self.rotate_mag(
            self.mag_extracted,
            self.hopping,
            self.soc,
            axis_vec0,
            axis_vec1,
            theta,
            phi,
            lam_para,
        )

        if write_rotated_hr or (file_name != "pwscf_rot_hr.dat"):
            self.write_rotated_hr(
                self.num_wann,
                self.nrpts,
                self.ndegen,
                self.irvec,
                self.hrs,
                file_name=file_name,
                hr_dat=hr_dat,
                tb_dat=tb_dat,
            )

    def reshape_hrs_to_ham_spinor(self, hrs: np.ndarray) -> np.ndarray:
        """transform the non-colinear Hamiltonian hrs into the spinor form ham_spinor.

        Args:
            hrs: Non-collinear Hamiltonian in (nrpts, num_wann, num_wann) array.

        Return:
            array(complex)[:nrpts, :2, :2, :num_wann/2, :num_wann/2]
        """
        ham_spinor = hrs.reshape(
            self.nrpts, int(self.num_wann / 2), 2, int(self.num_wann / 2), 2
        )
        ham_spinor = np.einsum("ijklm -> ikmjl", ham_spinor)
        return ham_spinor

    def reshape_ham_spinor_to_hrs(self, ham_spinor: np.ndarray) -> np.ndarray:
        """transform the spinor form ham_spinor into the non-colinear Hamiltonian hrs.

        Args:
            ham_spinor: Spinor form Hamiltonian in (nrpts, 2, 2, num_wann/2, num_wann/2) array.

        Return:
            array(complex)[:nrpts, :num_wann, :num_wann]
        """
        hrs = np.einsum("ijklm -> iljmk", ham_spinor)
        hrs = hrs.reshape(self.nrpts, int(self.num_wann), int(self.num_wann))
        return hrs

    def su2_rotation(self, n: list, theta: float) -> np.ndarray:
        """create a representation of su2 rotation with spin 1/2, rotation axis n, and rotation angle theta.

        Args:
            n (array[:3]): Rotation axis.
            theta (float): Rotation angle [rad].

        Return:
            array(complex)[:2l+1, :2l+1]
        """
        sigma_x, sigma_y, sigma_z = (
            np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]]),
            np.array([[0.0 + 0.0j, 0.0 - 1.0j], [0.0 + 1.0j, 0.0 + 0.0j]]),
            np.array([[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]),
        )
        n = n / np.linalg.norm(n)
        rotation_matrix = expm(
            -1j * theta / 2 * (n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z)
        )
        return rotation_matrix

    def spinor_to_vector(self, spinor: np.ndarray) -> np.ndarray:
        """convert spinors to vectors in a Pauli matrix basis.

        Args:
            spinor: Spinor form array in (2, 2, num_wann/2, num_wann/2).

        Return:
            array[:4, :num_wann/2, :num_wann/2]
        """
        vector = np.zeros([4, spinor.shape[2], spinor.shape[3]], dtype=complex)
        vector[0, :, :] = 1 / 2 * (spinor[0, 0, :, :] + spinor[1, 1, :, :])
        vector[1, :, :] = 1 / 2 * (spinor[0, 1, :, :] + spinor[1, 0, :, :])
        vector[2, :, :] = 1 / (2j) * (spinor[1, 0, :, :] - spinor[0, 1, :, :])
        vector[3, :, :] = 1 / 2 * (spinor[0, 0, :, :] - spinor[1, 1, :, :])
        return vector

    def vector_to_spinor(self, vector: np.ndarray) -> np.ndarray:
        """convert vectors in a Pauli matrix basis to spinors.

        Args:
            vector: Vector expanded by the Pauli matrix in (4, num_wann/2, num_wann/2).

        Return:
            array[:2, :2, :num_wann/2, :num_wann/2]
        """
        spinor = np.zeros([2, 2, vector.shape[1], vector.shape[2]], dtype=complex)
        spinor[0, 0, :, :] = vector[0, :, :] + vector[3, :, :]
        spinor[1, 1, :, :] = vector[0, :, :] - vector[3, :, :]
        spinor[0, 1, :, :] = vector[1, :, :] - 1j * vector[2, :, :]
        spinor[1, 0, :, :] = vector[1, :, :] + 1j * vector[2, :, :]
        return spinor

    def split_ham(
        self, ham_r: HamR, extract_only_x_component: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """divide the Hamiltonian into hopping, magnetization, and SOC.

        The magnetization mag_orig is taken as the term antisymmetric to time reversal in the Hamiltonian,
        and only the x component of the magnetization mag_extracted is taken if needed.
        Among the time-reversal symmetric terms in the Hamiltonian,
        we separate hopping as the spin-independent component
        and soc as the spin-dependent component.

        Args:
            ham_r (HamR): HamR instance.
            extract_only_x_component (bool): Whether to extract x component of mag.

        Returns:
            tuple[:4]: mag_orig, mag_extracted, hopping, and soc. Each in (nrpts, num_wann, num_wann) array.
        """
        ham_spinor = self.reshape_hrs_to_ham_spinor(ham_r.hrs)
        unitary_tr = self.su2_rotation([0, 1, 0], np.pi)
        ham_tr = np.einsum(
            "ij,ojklm,kn->oinlm",
            unitary_tr,
            np.conjugate(ham_spinor),
            np.conjugate(unitary_tr.T),
        )
        mag_orig = 1 / 2 * (ham_spinor - ham_tr)
        mag_extracted = mag_orig.copy()
        if extract_only_x_component:
            for i in range(self.nrpts):
                _mag_vec = self.spinor_to_vector(mag_orig[i, :, :, :, :])
                _mag_x = np.zeros(_mag_vec.shape, dtype=complex)
                _mag_x[1, :, :] = _mag_vec[1, :, :]
                mag_extracted[i, :, :, :, :] = self.vector_to_spinor(_mag_x)

        ham_sym = 1 / 2 * (ham_spinor + ham_tr)
        hopping, soc = ham_spinor.copy(), np.zeros(ham_spinor.shape, dtype=complex)
        for i in range(self.nrpts):
            _ham_sym_vec = self.spinor_to_vector(ham_sym[i, :, :, :, :])
            _hopping_vec, _soc_vec = (
                np.zeros(_ham_sym_vec.shape, dtype=complex),
                np.zeros(_ham_sym_vec.shape, dtype=complex),
            )
            _hopping_vec[0, :, :] = _ham_sym_vec[0, :, :]
            _soc_vec[1:4, :, :] = _ham_sym_vec[1:4, :, :]
            hopping[i, :, :, :, :] = self.vector_to_spinor(_hopping_vec)
            soc[i, :, :, :, :] = self.vector_to_spinor(_soc_vec)
        return (
            self.reshape_ham_spinor_to_hrs(mag_orig),
            self.reshape_ham_spinor_to_hrs(mag_extracted),
            self.reshape_ham_spinor_to_hrs(hopping),
            self.reshape_ham_spinor_to_hrs(soc),
        )

    def rotate_mag(
        self,
        mag: np.ndarray,
        hopping: np.ndarray,
        soc: np.ndarray,
        axis_vec0: list,
        axis_vec1: list,
        theta: float,
        phi: float,
        lam_para: float,
    ) -> np.ndarray:
        """rotate the magnetization and create a new Hamiltonian.
        Two rotations are performed to reconstruct a Hamiltonian
        that depends on the direction of magnetization (theta, phi).

        Args:
            mag: Magnetization in (nrpts, num_wann, num_wann) array.
            hopping: Hopping in (nrpts, num_wann, num_wann) array.
            soc: SOC in (nrpts, num_wann, num_wann) array.
            axis_vec0 (list): 1st rotation axis.
            axis_vec1 (list): 2nd rotation axis.
            theta (float): 1st rotation angle [rad].
            phi (float): 2nd rotation angle [rad].
            lam_para (float): Parameter of the SOC magnitude.

        Returns:
            hrs: Hamiltonian after mag rotation in (nrpts, num_wann, num_wann) array.
        """
        unitary_0 = self.su2_rotation(axis_vec0, theta)
        unitary_1 = self.su2_rotation(axis_vec1, phi)

        mag_0 = np.einsum(
            "ij,ojklm,kn->oinlm",
            unitary_0,
            self.reshape_hrs_to_ham_spinor(mag),
            np.conjugate(unitary_0.T),
        )
        mag_1 = np.einsum(
            "ij,ojklm,kn->oinlm", unitary_1, mag_0, np.conjugate(unitary_1.T)
        )
        hrs = self.reshape_ham_spinor_to_hrs(mag_1) + lam_para * soc + hopping
        return hrs

    def write_rotated_hr(
        self,
        num_wann: int,
        nrpts: int,
        ndegen: List,
        irvec: np.ndarray,
        hrs: np.ndarray,
        file_name: str,
        hr_dat: Optional[Path] = None,
        tb_dat: Optional[Path] = None,
    ):
        """write a hr.dat file of Hamiltonian with rotated magnetization.

        Args:
        num_wann (int): Number of Wannier orbitals.
        nrpts (int): Number of unit cells.
        ndegen: Degeneracies of Wigner-Seitz grid points in (nrpts) array.
        irvec: Lattice vectors for unit cells in (nrpts, 3) array.
        hrs: Wannier-based Hamiltonian in (nrpts, num_wann, num_wann) array.
        file_name (str, optional): Name of rotated_hr.
        hr_dat: wannier_hr.dat file path. defaults to None.
        tb_dat: wannier_tb.dat file path. defaults to None.
        """
        if tb_dat:
            parent_path = tb_dat.parent
        elif hr_dat:
            parent_path = hr_dat.parent
        else:
            raise ValueError("Neither tb_dat nor hr_dat is given.")

        now = datetime.now()
        date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(parent_path / str(file_name), "w") as f:
            f.write("written by mag_rotation.py at {}\n".format(date_time_str))
            f.write("{}\n".format(num_wann))
            f.write("{}\n".format(nrpts))
            for i in range(0, len(ndegen), 15):
                line = "\t".join(map(str, ndegen[i : i + 15]))
                f.write(line + "\n")
            for j, (m, n) in product(range(nrpts), product(range(num_wann), repeat=2)):
                f.write("\t".join(map(str, irvec[j])) + "\t")
                f.write("\t".join(map(str, [n, m])) + "\t")
                f.write(
                    "\t".join(
                        map(
                            str,
                            [
                                "{:.8e}".format(hrs[j, n, m].real),
                                "{:.8e}".format(hrs[j, n, m].imag),
                            ],
                        )
                    )
                    + "\n"
                )
