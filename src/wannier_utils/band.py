#!/usr/bin/env python

from itertools import chain
from typing import Iterable, List

import numpy as np 
import matplotlib as mpl
mpl.use("cairo")
from matplotlib import pyplot as plt
from seekpath import get_explicit_k_path
from pymatgen.core.periodic_table import Element
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.bandstructure import HighSymmKpath

from wannier_utils.wannier_system import WannierSystem
from wannier_utils.logger import get_logger

logger = get_logger(__name__)


def main():
    from pathlib import Path
    from time import perf_counter

    base_dir = Path(__file__).parents[2]
    ws = WannierSystem(
        hr_dat=base_dir/"tests"/"Fe_hr.dat", 
        nnkp_file=base_dir/"tests"/"Fe.nnkp", 
        win_file=base_dir/"tests"/"Fe.win",
    )

    start_time = perf_counter()
    bs = BandStructure(ws, projection=True)
    bs.plot_bands_seekpath()
    end_time = perf_counter()
    logger.info(f"elapsed_time: {end_time - start_time} [sec]")


class BandStructure:
    """
    Band structure plot.
    """
    def __init__(
        self, 
        ws: WannierSystem, 
        is_projection: bool = False, 
        file_prefix: str = "band", 
    ):
        """
        Constructor.

        Args:
            ws: a WannierSystem instance.
            is_projection: whether to output eigenvectors projected to real space.
            file_prefix: prefix of files of band structure data and plotting script.
        """
        self.ws = ws
        self.is_projection = is_projection
        self.file_prefix = file_prefix

    def plot_bands(
        self, 
        kpts: np.ndarray, 
        kmeshes: Iterable[int], 
        klabels: Iterable[str] = [], 
    ):
        """
        Get band structure information manually.

        Args:
            kpts (np.ndarray): (high-symmetry) k points in fractional coordinates.
            kmeshes (Iterable[int]): the number of mesh of each (high-symmetry) k pathes.
            klabels (Iterable[str]): labels of each points in kpts. defaults to [].
        """
        kpts_all = [kpts[0]]
        kpts_lin = [0]
        tick_locs = [0]
        for i in range(len(kpts) - 1):
            kpts_all += [(kpts[i + 1] - kpts[i])*float(j + 1)/kmeshes[i] + kpts[i] \
                         for j in range(kmeshes[i])]
            d = np.linalg.norm(np.dot(kpts[i + 1] - kpts[i], self.ws.b), ord=2)
            kpts_lin += [d*float(j + 1)/kmeshes[i] + kpts_lin[-1] for j in range(kmeshes[i])]
            tick_locs.append(tick_locs[-1] + d)

        self._plot_bands(np.array(kpts_all), np.array(kpts_lin), tick_locs, klabels)

    def _plot_bands(
        self, 
        kpoints_frac: np.ndarray, 
        kpoints_lin: np.ndarray, 
        tick_locs: Iterable[float], 
        tick_labels: Iterable[str], 
    ):
        """
        internal routine called by plot_bands methods in this class.
        outputting band energies and weights to prefix.dat and band structure in prefix.pdf.

        Args:
            kpoints_frac: 3D k points in fractional coordinates to get energy eigenvalues.
            kpoints_lin: 1D values to plot bands.
            tick_locs: values in kpoints_lin corresponding to high-symmetry points.
            tick_labels: labels of high-symmetry points in tick_locs.
        """
        ham_ks = [self.ws.calc_ham_k(kf, diagonalize=True) for kf in kpoints_frac]
        bands = np.array([ham_k.ek for ham_k in ham_ks])

        mpl.rcParams["text.usetex"] = True
        mpl.rcParams["text.latex.preamble"] = "\\usepackage{amssymb} \\usepackage{amsmath}"
        ax = plt.subplot()
        ax.plot(kpoints_lin, bands, ls="-", lw=1, c="k")
        y_min = bands.min() - 0.025*(bands.max() - bands.min())
        y_max = bands.max() + 0.025*(bands.max() - bands.min())
        ax.set_ylim([y_min, y_max])
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Energy [eV]")
        for loc, label in zip(tick_locs, tick_labels):
            lw = 1 if "/" in label else 0.5
            ax.axvline(loc, color="k", ls="-", lw=lw)
        plt.savefig(self.file_prefix + ".pdf", bbox_inches="tight")

    def plot_bands_seekpath(self):
        st = self.ws.win.structure
        cell = st.lattice.matrix
        positions = st.frac_coords
        numbers = [Element(s).Z for s in st.species]
        kpath = get_explicit_k_path((cell, positions, numbers))

        new_b = kpath["reciprocal_primitive_lattice"]
        m = np.matmul(new_b, cell.T)/(2*np.pi)
        kpoints_frac = [np.matmul(k, m) for k in kpath["explicit_kpoints_rel"]]   # convert to original cell
        kpoints_lin = kpath["explicit_kpoints_linearcoord"]

        kpoints_labels = kpath["explicit_kpoints_labels"]
        tick_locs, tick_labels = [], []
        for i, l in enumerate(kpoints_labels):
            if not l: continue
            l = l.replace("GAMMA", "$\\Gamma$")
            l = l.replace("SIGMA", "$\\Sigma$")
            if ("_" in l):
                label = label.replace("_", "$_") + "$"

            if i and kpoints_labels[i - 1]:
                tick_labels[-1] += "/" + l
            else:
                tick_locs.append(kpoints_lin[i])
                tick_labels.append(l)

        self._plot_bands(kpoints_frac, kpoints_lin, tick_locs, tick_labels)

    def plot_bands_pymatgen(self, line_density: int = 50):
        st = self.ws.win.structure
        has_magmoms = "magmom" in st.site_properties.keys()
        if has_magmoms:
            magmoms = st.site_properties["magmom"]
            if Magmom.have_consistent_saxis(magmoms):
                magmom_saxis = magmoms[0].saxis
            else:
                magmoms, saxis = Magmom.get_consistent_set_and_saxis(magmoms)
                st.site_properties["magmom"] = magmoms
                magmom_saxis = saxis
        else:
            magmom_saxis = None
        hsk = HighSymmKpath(st, has_magmoms=has_magmoms, magmom_axis=magmom_saxis)
        kpoints_cart, klabels = hsk.get_kpoints(line_density=line_density)
        kpoints_lin = self.get_kpoints_lin(kpoints_cart, klabels)
        kpoints_frac = [
            st.lattice.reciprocal_lattice.get_fractional_coords(k) for k in kpoints_cart
        ]

        # eliminate continuous overlaps in kpoints and klabels
        klabels_nolap, kpoints_lin_nolap, kpoints_frac_nolap  = [], [], []
        for i, (kl, kpl, kpf) in enumerate(zip(klabels, kpoints_lin, kpoints_frac)):
            if i and kl and kl == klabels[i - 1]:
                continue
            klabels_nolap.append(kl)
            kpoints_lin_nolap.append(kpl)
            kpoints_frac_nolap.append(kpf)

        tick_locs, tick_labels = [], []
        for i, l in enumerate(klabels_nolap):
            if not l: continue
            l = l.replace("\Gamma", "$\\Gamma$")
            l = l.replace("\Sigma", "$\\Sigma$")

            if i and klabels_nolap[i - 1]:
                tick_labels[-1] += "/" + l
            else:
                tick_locs.append(kpoints_lin[i])
                tick_labels.append(l)

        self._plot_bands(kpoints_frac_nolap, kpoints_lin_nolap, tick_locs, tick_labels)

    @staticmethod
    def get_kpoints_lin(kpoints_cart: np.ndarray, klabels: Iterable[str]) -> List[float]:
        kpoints_cart_sep, kpoints_tmp = [], []
        for k, l in zip(kpoints_cart, klabels):
            if not kpoints_tmp and l:
                kpoints_tmp = [k]
                continue
            kpoints_tmp.append(k)
            if kpoints_tmp and l:
                kpoints_cart_sep.append(kpoints_tmp)
                kpoints_tmp = []
        norms_sep = []
        for i, ks in enumerate(kpoints_cart_sep):
            ns = [np.linalg.norm(k - ks[0], ord=2) for k in ks]
            if i:
                ns = [n + norms_sep[-1][-1] for n in ns]
            norms_sep.append(ns)
        return list(chain.from_iterable(norms_sep))


if __name__ == "__main__":
    main()
