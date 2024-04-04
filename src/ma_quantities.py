# -*- coding: utf-8 -*-

"""Module to calculate physical quantities dependent on the direction of magnetization.

Todo:
    * 同じ元素が複数ある場合に、 orbitals がバグっている。

Example:
    Write the results of the physical quantity calculation in mp_folder/result.toml.
    >>> cd path/to/input_params.toml
    >>> mpirun -n 8 python ma_quantities.py
"""
from pathlib import Path
from typing import Optional, Tuple
import os
import numpy as np
import h5py
import toml

from mag_rotation import MagRotation
from parallel_eigval import ParallelEigval
from wannier_utils.hamiltonian import HamK
from wannier_utils.mp_points import MPPoints
from wannier_utils.wannier_system import WannierSystem


class MaQuantities:
    """From crystal data obtained with WannierSystem and real-space Hamiltonian obtained with HamR,
    this class calculates the total energy, spin angular momentum, orbital angular momentum.

    Args:
        ham_r (HamR or MagRotation): Real space Hamiltonian object.
        num_valence (int): Number of valence electrons in DFT.
        win_file (Path): Path to pwscf.win.
        kmesh (list[:3], optional): k-mesh fineness in x,y,z directions. Default to [10,10,10].
        work_dir (Path, optional): Path to make a work dir to write and read temporary files. Default to None.
        calc_spin_angular_momentum (bool, optional): Whether to calculate spin angular momentum. Default to False.
        calc_orbital_angular_momentum (bool, optional): Whether to calculate orbital angular momentum. Default to False.

    Attributes:
        ws (WannierSystem): WannierSystem object.
        orbitals (list(str)): List of wannier orbitals.
        sorted_energy (array[:num_wann * klen]): List of eigenvalues sorted by increasing order.
        sorted_eigvec (array(complex)[:num_wann, :num_wann * klen]): List of eigenvecs sorted by increasing order.
        fermi_energy (float): Fermi energy [eV].
        free_energy (float): Free energy [eV].
        spin_angular_momentum (array): Spin angular momentum [mu_B].
        orbital_angular_momentum (array): Orbital angular momentum [mu_B].

    Notes:
        Using temporary files is slower but consumes less memory; recommended for dense k-point calculations.
    """

    def __init__(
        self,
        ham_r: MagRotation,
        num_valence: int,
        win_file: Path,
        kmesh: Optional[list] = [10, 10, 10],
        work_dir: Optional[Path] = None,
        calc_spin_angular_momentum: Optional[bool] = False,
        calc_orbital_angular_momentum: Optional[bool] = False,
    ):
        """Constructor that reads crystal data in WannierSystem
        and makes a uniform k-point list in MPPoints,
        Fourier-transforms and diagonalizes ham_r in HamK,
        sorts and writes out the eigvals and eigvecs in order of increasing energy,
        and reads it to calculate physical quantities.
        """
        self.ws = WannierSystem(win_file=win_file)
        mp_points = MPPoints(self.ws, kmesh)
        klist = mp_points._get_full_kvec()
        self.klen = np.shape(klist)[0]
        self.num_wann = ham_r.num_wann
        self.calc_spin_angular_momentum = calc_spin_angular_momentum
        self.calc_orbital_angular_momentum = calc_orbital_angular_momentum

        energy, eigvec = self.diag_in_all_k(ham_r, klist)
        self.sorted_energy, self.sorted_eigvec = self.sort_energy(energy, eigvec)
        self.fermi_energy, self.free_energy = self.calc_fermi_and_free_energy(
            self.sorted_energy, num_valence
        )

        if work_dir:
            self._write_eigvals_and_eigvecs(work_dir)
        if calc_spin_angular_momentum:
            self.spin_angular_momentum = self._calc_spin_angular_momentum(
                num_valence, work_dir=work_dir
            )
        self.orbitals = self.make_orbital_list(self.ws)
        if calc_orbital_angular_momentum:
            self.orbital_angular_momentum = self._calc_orbital_angular_momentum(
                num_valence, self.orbitals, work_dir=work_dir
            )

    def _diag(self, ki, ham_r: MagRotation, klist):
        """used for multiprocessing."""
        ham_k = HamK(ham_r, klist[ki, :], diagonalize=True)
        return ham_k.ek

    def diag_in_all_k(
        self, ham_r: MagRotation, klist: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """calculate energy eigenvalues and eigenstates for all k points in klist.

        Args:
            ham_r (HamR or MagRotation): Real space Hamiltonian object.
            klist: List of kpoints in (klen, 3) array.

        Returns:
            energy: List of eigvals in (num_wann, klen) array.
            eigvec: List of eigvecs in (num_wann, num_wann, klen) array.
        """
        if self.calc_spin_angular_momentum or self.calc_orbital_angular_momentum:
            energy = np.empty((self.num_wann, self.klen), dtype=float)
            eigvec = np.empty((self.num_wann, self.num_wann, self.klen), dtype=complex)
            for ki in range(self.klen):
                ham_k = HamK(ham_r, klist[ki, :], diagonalize=True)
                energy[:, ki], eigvec[:, :, ki] = ham_k.ek, ham_k.uk
        else:
            # Multi-processing to calculate only eigvals.
            parallel = ParallelEigval(ham_r, klist)
            energy = parallel.energy
            eigvec = None
        return energy, eigvec

    def sort_energy(
        self, energy: np.ndarray, eigvec: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """sort eigenvalues and eigenstates in increasing order of energy eigenvalue.

        Args:
            energy: List of eigvals in (num_wann, klen) array.
            eigvec: List of eigvecs in (num_wann, num_wann, klen) array.

        Returns:
            sorted_energy: List of energy eigenvalues after sorting in (num_wann * klen) array.
            sorted_eigvec: List of energy eigenstates after sorting in (num_wann, num_wann * klen) array.
        """
        idx = np.argsort(energy.reshape(self.num_wann * self.klen), kind="heapsort")
        sorted_energy = energy.reshape(self.num_wann * self.klen)[idx]
        sorted_eigvec = None
        if self.calc_spin_angular_momentum or self.calc_orbital_angular_momentum:
            sorted_eigvec = eigvec.reshape(self.num_wann, self.num_wann * self.klen)[
                :, idx
            ]
        return sorted_energy, sorted_eigvec

    def calc_fermi_and_free_energy(
        self, sorted_energy: np.ndarray, num_valence: int
    ) -> Tuple[float, float]:
        """calculate Fermi energy and free energy.

        Args:
            sorted_energy: List of energy eigenvalues after sorting in (num_wann * klen) array.
            num_valence (int): Number of valence electrons in DFT.

        Returns:
            tuple[:2]: Tuple of Fermi energy and free energy.
        """
        fermi_energy = sorted_energy[num_valence * self.klen - 1]
        free_energy = np.sum(sorted_energy[0 : (num_valence * self.klen)]) / self.klen
        return fermi_energy, free_energy

    def _write_eigvals_and_eigvecs(self, work_dir: Path):
        """write the eigenvalues and eigenvectors at each k-point to a file under work_dir/work/.

        Notes:
            sorted_energy.txt: Text file of sorted eigenvalues.
            sorted_eigvec.hdf5: Binary (hdf5 format) file of sorted eigenvectors.
        """
        try:
            os.mkdir(work_dir / "work")
        except:
            pass

        f = open(work_dir / "work" / "energy.txt", "w")
        f.close()
        for i in range(len(self.sorted_energy)):
            f = open(work_dir / "work" / "energy.txt", "a")
            f.write(str(self.sorted_energy[i]) + "\n")
            f.close()

        f = h5py.File(work_dir / "work" / "eigvec.hdf5", "w")
        f.create_dataset("sorted_eigvec", data=self.sorted_eigvec)
        f.close()

    def _calc_spin_angular_momentum(
        self, num_valence: int, work_dir: Path = None,
    ) -> np.ndarray:
        """calculate spin angular momentum.

        Args:
            num_valence (int): Number of valence electrons in DFT.
            work_dir (Path): Whether to use temporary files.

        Return:
            array[:3]: Spin angular momentum [mu_B].
        """
        sigma = self.lie_algebra(1 / 2)
        if work_dir:
            spin_angular_momentum = np.zeros(3)
            f = h5py.File(work_dir / "work" / "eigvec.hdf5", "r")
            for ki in range(num_valence * self.klen):
                _eigvec_k = f["sorted_eigvec"][:, ki]
                _eigvec_k = _eigvec_k.reshape(int(self.num_wann / 2), 2)
                spin_angular_momentum[:] += (
                    np.einsum(
                        "ij,mjl,il -> m", np.conjugate(_eigvec_k), sigma, _eigvec_k
                    )
                    / self.klen
                ).real
            f.close()
        else:
            _eigvec = self.sorted_eigvec.reshape(
                int(self.num_wann / 2), 2, self.num_wann * self.klen
            )
            spin_angular_momentum = (
                np.einsum(
                    "ijk,mjl,ilk -> m",
                    np.conjugate(_eigvec[:, :, : num_valence * self.klen]),
                    sigma,
                    _eigvec[:, :, : num_valence * self.klen],
                )
                / self.klen
            ).real
        return spin_angular_momentum

    def lie_algebra(self, l: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """create a representation of the Lie algebra of Casimir l.
        The bases are ordered in order of decreasing weight.

        Args:
            l (float): integer or half-integer.

        Returns:
            tuple[:3]: Tuple of Lie algebra, sigma_i in (2l+1, 2l+1) array.
        """
        n = int(2 * l + 1)
        sigma_x = np.zeros([n, n], dtype=complex)
        sigma_y = np.zeros([n, n], dtype=complex)
        sigma_z = np.zeros([n, n], dtype=complex)
        for i in range(n):
            for j in range(n):
                if i == (j + 1):
                    sigma_x[i, j] = np.sqrt((l + 1) * (i + j + 1) - (i + 1) * (j + 1))
                    sigma_y[i, j] = 1j * np.sqrt(
                        (l + 1) * (i + j + 1) - (i + 1) * (j + 1)
                    )
                if (i + 1) == j:
                    sigma_x[i, j] = np.sqrt((l + 1) * (i + j + 1) - (i + 1) * (j + 1))
                    sigma_y[i, j] = -1j * np.sqrt(
                        (l + 1) * (i + j + 1) - (i + 1) * (j + 1)
                    )
                if i == j:
                    sigma_z[i, j] = 2 * (l - i)
        return sigma_x, sigma_y, sigma_z

    def make_orbital_list(self, ws: WannierSystem) -> list:
        """create a list of orbits from a proj_list of WannierSystem.

        Args:
            ws (WannierSystem): a WannierSystem instance.

        Returns:
            list(str): List of Wannier orbitals.
        """
        orbitals = []
        for line in ws.win.proj_list:
            line = line.split(":")[1]
            for char in line:
                if char == "s":
                    orbitals.append(char)
                if char == "p":
                    orbitals.append(char)
                if char == "d":
                    orbitals.append(char)
                if char == "f":
                    orbitals.append(char)
        return orbitals

    def _calc_orbital_angular_momentum(
        self, num_valence: int, orbitals: list, work_dir: Path = None,
    ) -> np.ndarray:
        """calculate orbital angular momentum.

        Args:
            num_valence (int): Number of valence electrons in DFT.
            orbitals (list): List of Wannier orbitals.
            work_dir (Path): Whether to use temporary files.

        Retuern:
            array[:3]: Orbital angular momentum [mu_B]

        Notes:
            transform_p, transform_d, transform_f: A transformation matrix that rearranges the orbits in order of increasing weight to the Wannier90 standard.
            orbital_rep: Angular momentum operator expressed in Wannier basis in (3, num_wann, num_wann). Only the spin space diagonal component has a value.
        """
        sigma_p = self.lie_algebra(1)
        sigma_d = self.lie_algebra(2)
        sigma_f = self.lie_algebra(3)
        transform_p = (
            1
            / np.sqrt(2)
            * np.array([[0, np.sqrt(2), 0], [-1, 0, 1], [1j, 0, 1j]], dtype=complex)
        )
        transform_d = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    [0, 0, np.sqrt(2), 0, 0],
                    [0, -1, 0, 1, 0],
                    [0, 1j, 0, 1j, 0],
                    [1, 0, 0, 0, 1],
                    [-1j, 0, 0, 0, 1j],
                ],
                dtype=complex,
            )
        )
        transform_f = (
            1
            / np.sqrt(2)
            * np.array(
                [
                    [0, 0, 0, np.sqrt(2), 0, 0, 0],
                    [0, 0, -1, 0, 1, 0, 0],
                    [0, 0, 1j, 0, 1j, 0, 0],
                    [0, 1, 0, 0, 0, 1, 0],
                    [0, -1j, 0, 0, 0, 1j, 0],
                    [-1, 0, 0, 0, 0, 0, 1],
                    [1j, 0, 0, 0, 0, 0, 1j],
                ],
                dtype=complex,
            )
        )
        _orbital_rep_spinor = np.zeros(
            [3, int(self.num_wann / 2), 2, int(self.num_wann / 2), 2], dtype=complex
        )
        idx = 0
        for orb in orbitals:
            if orb == "s":
                idx += 1
            if orb == "p":
                for spin_idx in range(2):
                    _orbital_rep_spinor[
                        :, idx : idx + 3, spin_idx, idx : idx + 3, spin_idx
                    ] = np.einsum(
                        "ij,kjl,lm -> kim",
                        transform_p,
                        sigma_p,
                        np.conjugate(transform_p.T),
                    )
                idx += 3
            if orb == "d":
                for spin_idx in range(2):
                    _orbital_rep_spinor[
                        :, idx : idx + 5, spin_idx, idx : idx + 5, spin_idx
                    ] = np.einsum(
                        "ij,kjl,lm -> kim",
                        transform_d,
                        sigma_d,
                        np.conjugate(transform_d.T),
                    )
                idx += 5
            if orb == "f":
                for spin_idx in range(2):
                    _orbital_rep_spinor[
                        :, idx : idx + 7, spin_idx, idx : idx + 7, spin_idx
                    ] = np.einsum(
                        "ij,kjl,lm -> kim",
                        transform_f,
                        sigma_f,
                        np.conjugate(transform_f.T),
                    )
                idx += 7
        if idx is not int(self.num_wann / 2):
            raise ValueError("Orbitals are strange.")

        orbital_rep = _orbital_rep_spinor.reshape(3, self.num_wann, self.num_wann)
        orbital_angular_momentum = np.zeros(3)
        if work_dir:
            f = h5py.File(work_dir / "work" / "eigvec.hdf5", "r")
            for ki in range(num_valence * self.klen):
                _eigvec_k = f["sorted_eigvec"][:, ki]
                orbital_angular_momentum[:] += (
                    np.einsum(
                        "i,kij,j -> k", np.conjugate(_eigvec_k), orbital_rep, _eigvec_k
                    )
                    / self.klen
                ).real
            f.close()
        else:
            _eigvec = self.sorted_eigvec
            orbital_angular_momentum = (
                np.einsum(
                    "jk,mjl,lk -> m",
                    np.conjugate(_eigvec[:, : num_valence * self.klen]),
                    orbital_rep,
                    _eigvec[:, : num_valence * self.klen],
                )
                / self.klen
            ).real
        return orbital_angular_momentum


if __name__ == "__main__":
    with open("input_params.toml", "r") as f:
        params = toml.load(f)

    hr_dat, tb_dat = None, None
    if "hr_dat" in params.keys():
        hr_dat = Path(params["hr_dat"])
    if "tb_dat" in params.keys():
        tb_dat = Path(params["tb_dat"])
    win_file = Path(params["win_file"])
    nnkp_file = Path(params["nnkp_file"])
    mp_folder = Path(params["mp_folder"])

    ham_z = MagRotation(
        hr_dat=hr_dat,
        tb_dat=tb_dat,
        extract_only_x_component=int(params["extract_only_x_component"]),
        use_convert_ham_r=int(params["use_convert_ham_r"]),
        nnkp_file=nnkp_file,
        win_file=win_file,
    )
    ham_x = MagRotation(
        hr_dat=hr_dat,
        tb_dat=tb_dat,
        extract_only_x_component=int(params["extract_only_x_component"]),
        use_convert_ham_r=int(params["use_convert_ham_r"]),
        nnkp_file=nnkp_file,
        win_file=win_file,
        theta=np.pi / 2,
    )
    Nk = int(params["Nk"])
    kmesh = [Nk, Nk, Nk]
    num_valence = int(params["num_valence"])
    mae_z = MaQuantities(
        ham_z,
        num_valence,
        kmesh=kmesh,
        win_file=win_file,
        # work_dir=mp_folder,
        # calc_spin_angular_momentum=True,
        # calc_orbital_angular_momentum=True,
    )
    mae_x = MaQuantities(
        ham_x,
        num_valence,
        kmesh=kmesh,
        win_file=win_file,
        # work_dir=mp_folder,
    )

    result_contents = {
        "atoms": mae_z.ws.win.atom_list,
        "orbitals": mae_z.orbitals,
        "extract_only_x_component": params["extract_only_x_component"],
        "use_convert_ham_r": params["use_convert_ham_r"],
        "Nk": Nk,
        "E_F [eV]": float(mae_z.fermi_energy),
        "dE[100] [meV]": float((mae_x.free_energy - mae_z.free_energy) * 10 ** 3),
        "M_diff_max [meV]": float(
            np.max(np.abs(ham_z.mag_extracted - ham_z.mag_orig)) * 10 ** 3
        ),
        "M_diff_ave [meV]": float(
            np.mean(np.abs(ham_z.mag_extracted - ham_z.mag_orig)) * 10 ** 3
        ),
        # "spin_moment [mu_B]": [float(_) for _ in mae_z.spin_angular_momentum],
        # "orbital_moment [mu_B]": [float(_) for _ in mae_z.orbital_angular_momentum],
    }
    with open(mp_folder / "result.toml", "w") as f:
        toml.dump(result_contents, f)
