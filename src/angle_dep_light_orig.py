# -*- coding: utf-8 -*-

"""maeの角度依存性を(phi, theta) = (0, theta)上と、(phi, theta) = (phi, pi/2)上でのみ計算する。

Example:
    >>> python src/angle_dep.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import minimize
import toml

from mag_rotation import MagRotation
from ma_quantities import MaQuantities


class Angle_dep:
    def __init__(self):
        self.calc()

    def func(self, angle, params):
        """ある角度phi, theta [rad]でのsorted_energyを計算する関数。"""
        theta, phi = angle
        ham = MagRotation(
            tb_dat=Path(params["tb_dat"]),
            nnkp_file=Path(params["nnkp_file"]),
            win_file=Path(params["win_file"]),
            extract_only_x_component=int(params["extract_only_x_component"]),
            use_convert_ham_r=int(params["use_convert_ham_r"]),
            theta=theta,
            phi=phi,
        )
        mae = MaQuantities(
            ham,
            int(params["num_valence"]),
            kmesh=[int(params["Nk"]), int(params["Nk"]), int(params["Nk"])],
            win_file=Path(params["win_file"]),
        )
        return mae.free_energy

    def calc(self):
        """phi 45, theta 45 の4度刻みのメッシュでエネルギーを計算し、結果を angle_dep.txt に書き込む。"""
        with open("input_params.toml", "r") as f:
            params = toml.load(f)

        with open(params["angle_dep_file_name"], "w") as f:
            f.write("Nk = {}\n".format(params["Nk"]))
            f.write("E [eV]\n")
            f.write("(phi, theta) = (0, theta)\n")

        for t_i in np.linspace(0, np.pi, 45):
            energy = self.func((t_i, 0), params)
            if np.any(energy != None):
                with open(params["angle_dep_file_name"], "a") as f:
                    f.write(str(energy) + "\n")

        with open(params["angle_dep_file_name"], "a") as f:
            f.write("(phi, theta) = (phi, pi/2)\n")

        for p_j in np.linspace(0, np.pi, 45):
            energy = self.func((np.pi / 2, p_j), params)
            if np.any(energy != None):
                with open(params["angle_dep_file_name"], "a") as f:
                    f.write(str(energy) + "\n")


if __name__ == "__main__":
    Angle_dep()
