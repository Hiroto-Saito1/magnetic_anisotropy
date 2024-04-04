#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""maeのフェルミエネルギーへの依存性を計算し、結果をプロットする。

Example:
    >>> python src/bandfilling.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import toml
import sys

sys.path.append("../../../src")

from mag_rotation import MagRotation
from ma_quantities import MaQuantities


class Bandfilling:
    def __init__(self):
        self.calc()
        # self.plot()

    def band_filling_energy(self, sorted_energy, num_valence: float) -> float:
        """エネルギー固有値と価電子数を引数に、自由エネルギーを計算する関数。"""
        Nk = 100
        klen = Nk ** 3
        free_energy = np.sum(sorted_energy[0 : int(num_valence * klen)]) / klen
        return free_energy

    def calc(self, L=1000):
        """num_valence-2 から num_valence+2 の範囲をLのメッシュで自由エネルギーの差を計算し、
        結果を bandfilling.txt に書き出す。
        1列目はextract_only_x_component=True, 2列目はextract_only_x_component=False。"""
        self.L = L
        num_valence = 18
        Nk = 100
        with open("./bandfilling.txt", "w") as f:
            f.write("Nk = {} \t extract=False \n".format(Nk))
        ham_z0 = MagRotation(
            tb_dat=Path("mae/wan/pwscf_py_tb.dat"),
            extract_only_x_component=False,
            use_convert_ham_r=False,
            win_file=Path("mae/wan/pwscf.win"),
            nnkp_file=Path("mae/wan/pwscf.nnkp"),
        )
        ham_x0 = MagRotation(
            tb_dat=Path("mae1/wan/pwscf_py_tb.dat"),
            extract_only_x_component=False,
            use_convert_ham_r=False,
            win_file=Path("mae/wan/pwscf.win"),
            nnkp_file=Path("mae/wan/pwscf.nnkp"),
        )
        mae_z0 = MaQuantities(
            ham_z0, num_valence, kmesh=[Nk, Nk, Nk], win_file=Path("mae/wan/pwscf.win"),
        )
        mae_x0 = MaQuantities(
            ham_x0, num_valence, kmesh=[Nk, Nk, Nk], win_file=Path("mae/wan/pwscf.win"),
        )

        band_filling = np.linspace(num_valence - 2, num_valence + 2, self.L)

        sorted_energy_z0 = mae_z0.sorted_energy
        sorted_energy_x0 = mae_x0.sorted_energy
        for i in range(self.L):
            free_energy_z0 = self.band_filling_energy(sorted_energy_z0, band_filling[i])
            free_energy_x0 = self.band_filling_energy(sorted_energy_x0, band_filling[i])
            with open("./bandfilling.txt", "a") as f:
                f.write("{} \n".format(free_energy_x0 - free_energy_z0))

    def plot(self):
        """bandfilling.txt の計算結果をプロットする。"""
        num_valence = 18
        self.L = 1000
        band_filling = np.linspace(num_valence - 2, num_valence + 2, self.L)
        data = pd.read_table("./bandfilling.txt")
        plt.plot(band_filling, data.iloc[:, 0] * 10 ** 3)
        plt.grid()
        plt.axvline(x=num_valence, color="grey")
        plt.xlabel("Bandfilling $q$")
        plt.ylabel("$E[100] - E[001]$ [meV]")
        plt.savefig("./bandfilling.pdf")


if __name__ == "__main__":
    Bandfilling()
