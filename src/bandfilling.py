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

from mag_rotation import MagRotation
from ma_quantities import MaQuantities


class Bandfilling:
    def __init__(self):
        self.calc()
        # self.plot()

    def band_filling_energy(self, sorted_energy, num_valence: float) -> float:
        """エネルギー固有値と価電子数を引数に、自由エネルギーを計算する関数。"""
        with open("input_params.toml", "r") as f:
            params = toml.load(f)
        klen = int(params["Nk"]) ** 3
        free_energy = np.sum(sorted_energy[0 : int(num_valence * klen)]) / klen
        return free_energy

    def calc(self, L=1000):
        """num_valence-2 から num_valence+2 の範囲をLのメッシュで自由エネルギーの差を計算し、
        結果を bandfilling.txt に書き出す。
        1列目はextract_only_x_component=True, 2列目はextract_only_x_component=False。"""
        self.L = L
        with open("input_params.toml", "r") as f:
            params = toml.load(f)
        with open("./bandfilling.txt", "w") as f:
            f.write("Nk = {} \t extract=True \t extract=False \n".format(params["Nk"]))
        ham_z1 = MagRotation(
            tb_dat=Path(params["tb_dat"]),
            nnkp_file=Path(params["nnkp_file"]),
            win_file=Path(params["win_file"]),
            extract_only_x_component=True,
            use_convert_ham_r=int(params["use_convert_ham_r"]),
        )
        ham_x1 = MagRotation(
            tb_dat=Path(params["tb_dat"]),
            nnkp_file=Path(params["nnkp_file"]),
            win_file=Path(params["win_file"]),
            extract_only_x_component=True,
            use_convert_ham_r=int(params["use_convert_ham_r"]),
            theta=np.pi / 2,
        )
        ham_z0 = MagRotation(
            tb_dat=Path(params["tb_dat"]),
            nnkp_file=Path(params["nnkp_file"]),
            win_file=Path(params["win_file"]),
            extract_only_x_component=False,
            use_convert_ham_r=int(params["use_convert_ham_r"]),
        )
        ham_x0 = MagRotation(
            tb_dat=Path(params["tb_dat"]),
            nnkp_file=Path(params["nnkp_file"]),
            win_file=Path(params["win_file"]),
            extract_only_x_component=False,
            use_convert_ham_r=int(params["use_convert_ham_r"]),
            theta=np.pi / 2,
        )
        mae_z1 = MaQuantities(
            ham_z1,
            int(params["num_valence"]),
            kmesh=[int(params["Nk"]), int(params["Nk"]), int(params["Nk"])],
            win_file=Path(params["win_file"]),
        )
        mae_x1 = MaQuantities(
            ham_x1,
            int(params["num_valence"]),
            kmesh=[int(params["Nk"]), int(params["Nk"]), int(params["Nk"])],
            win_file=Path(params["win_file"]),
        )
        mae_z0 = MaQuantities(
            ham_z0,
            int(params["num_valence"]),
            kmesh=[int(params["Nk"]), int(params["Nk"]), int(params["Nk"])],
            win_file=Path(params["win_file"]),
        )
        mae_x0 = MaQuantities(
            ham_x0,
            int(params["num_valence"]),
            kmesh=[int(params["Nk"]), int(params["Nk"]), int(params["Nk"])],
            win_file=Path(params["win_file"]),
        )

        band_filling = np.linspace(
            int(params["num_valence"]) - 2, int(params["num_valence"]) + 2, self.L
        )
        sorted_energy_z1 = mae_z1.sorted_energy
        sorted_energy_x1 = mae_x1.sorted_energy
        sorted_energy_z0 = mae_z0.sorted_energy
        sorted_energy_x0 = mae_x0.sorted_energy
        for i in range(self.L):
            free_energy_z1 = self.band_filling_energy(sorted_energy_z1, band_filling[i])
            free_energy_x1 = self.band_filling_energy(sorted_energy_x1, band_filling[i])
            free_energy_z0 = self.band_filling_energy(sorted_energy_z0, band_filling[i])
            free_energy_x0 = self.band_filling_energy(sorted_energy_x0, band_filling[i])
            with open("./bandfilling.txt", "a") as f:
                f.write(
                    "{} \t {} \n".format(
                        free_energy_x1 - free_energy_z1, free_energy_x0 - free_energy_z0
                    )
                )

    def plot(self):
        """bandfilling.txt の計算結果をプロットする。"""
        with open("input_params.toml", "r") as f:
            params = toml.load(f)
        band_filling = np.linspace(
            int(params["num_valence"]) - 2, int(params["num_valence"]) + 2, self.L
        )
        data = pd.read_table("./bandfilling.txt")
        plt.plot(band_filling, data.iloc[:, 0] * 10 ** 3, label="with extraction")
        plt.plot(band_filling, data.iloc[:, 1] * 10 ** 3, label="w/o extraction")
        plt.grid()
        plt.axvline(x=int(params["num_valence"]), color="grey")
        plt.legend()
        plt.xlabel("Bandfilling $q$")
        plt.ylabel("$E[100] - E[001]$ [meV]")
        plt.savefig("./bandfilling.pdf")


if __name__ == "__main__":
    Bandfilling()
