# -*- coding: utf-8 -*-

"""maeのSOIの大きさへの依存性を計算し、結果を lam_dep.txt に書き出す。

Example:
    >>> python src/lam_dep.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import toml

from mag_rotation import MagRotation
from ma_quantities import MaQuantities


class Lam_dep:
    def __init__(self):
        self.calc()

    def calc(self, L=40):
        """lambda = 0 から lamnda = 1.5 の範囲をLのメッシュで自由エネルギーの差を[eV]単位で計算し、
        結果を bandfilling.txt に書き出す。
        1列目はextract_only_x_component=True, 2列目はextract_only_x_component=False。"""
        with open("input_params.toml", "r") as f:
            params = toml.load(f)
        with open("./lam_dep.txt", "w") as f:
            f.write(
                "Nk = {} \nextract=True \t extract=False [eV]\n".format(params["Nk"])
            )

        for lam_para in np.linspace(0, 1.5, L):
            ham_z1 = MagRotation(
                tb_dat=Path(params["tb_dat"]),
                nnkp_file=Path(params["nnkp_file"]),
                win_file=Path(params["win_file"]),
                extract_only_x_component=True,
                use_convert_ham_r=int(params["use_convert_ham_r"]),
                lam_para=lam_para,
            )
            ham_x1 = MagRotation(
                tb_dat=Path(params["tb_dat"]),
                nnkp_file=Path(params["nnkp_file"]),
                win_file=Path(params["win_file"]),
                extract_only_x_component=True,
                use_convert_ham_r=int(params["use_convert_ham_r"]),
                theta=np.pi / 2,
                lam_para=lam_para,
            )
            ham_z0 = MagRotation(
                tb_dat=Path(params["tb_dat"]),
                nnkp_file=Path(params["nnkp_file"]),
                win_file=Path(params["win_file"]),
                extract_only_x_component=False,
                use_convert_ham_r=int(params["use_convert_ham_r"]),
                lam_para=lam_para,
            )
            ham_x0 = MagRotation(
                tb_dat=Path(params["tb_dat"]),
                nnkp_file=Path(params["nnkp_file"]),
                win_file=Path(params["win_file"]),
                extract_only_x_component=False,
                use_convert_ham_r=int(params["use_convert_ham_r"]),
                theta=np.pi / 2,
                lam_para=lam_para,
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

            with open("./lam_dep.txt", "a") as f:
                f.write(
                    "{} \t {} \n".format(
                        mae_x1.free_energy - mae_z1.free_energy,
                        mae_x0.free_energy - mae_z0.free_energy,
                    )
                )


if __name__ == "__main__":
    Lam_dep()
