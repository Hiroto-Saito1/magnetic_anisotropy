# -*- coding: utf-8 -*-

"""maeの角度依存性を計算し、最大値と最小値の角度を求め、最大値と最小値の角度を result.toml に書く。

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
        # self.energy = self.read()
        # self.optimize()

    def func(self, angle, params):
        """ある角度phi, theta での自由エネルギーを計算する関数。"""
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

        with open("angle_dep.txt", "w") as f:
            f.write("Nk = {}\n".format(params["Nk"]))
            f.write("E [eV]\n")
        phi = np.linspace(0, np.pi, 45)
        theta = np.linspace(0, np.pi, 45)
        for t_i in theta:
            for p_j in phi:
                energy = self.func((t_i, p_j), params)
                with open("angle_dep.txt", "a") as f:
                    if np.any(energy != None):
                        f.write(str(energy) + "\n")

    def read(self):
        """angle_dep.txt の結果を読み込み、行列の形に返す。"""
        energy = np.zeros([45, 45])
        f = open("angle_dep.txt", "r")
        f.readline()
        f.readline()
        for t_i in range(45):
            for p_j in range(45):
                energy[t_i, p_j] = float(f.readline())
        f.close()
        return energy

    def optimize(self):
        """全角度から、エネルギーの最大値と最小値を与える角度を見つける。"""
        with open("input_params.toml", "r") as f:
            params = toml.load(f)
        x0 = (0, 0)
        res_min = minimize(
            self.func, x0, args=(params), method="Nelder-Mead", tol=1e-5,
        )
        res_max = minimize(
            lambda *x: -self.func(*x),
            x0,
            args=(params),
            method="Nelder-Mead",
            tol=1e-5,
        )
        print(res_min)
        with open("result.toml", "r") as f:
            data = toml.load(f)
        data["min_angle(theta, phi)"] = [float(_ * 180 / np.pi) for _ in res_min.x]
        data["max_angle(theta, phi)"] = [float(_ * 180 / np.pi) for _ in res_max.x]
        data["dE_opt [meV]"] = float(-res_max.fun - res_min.fun) * 10 ** 3
        with open("result.toml", "w") as f:
            toml.dump(data, f)


if __name__ == "__main__":
    Angle_dep()
