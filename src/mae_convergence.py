# -*- coding: utf-8 -*-

"""input_params.toml を読み、磁気異方性エネルギーのk点数への収束を確認する。

Example:
    >>> python src/mae_convergence.py
"""

from pathlib import Path
import numpy as np
import toml

from mag_rotation import MagRotation
from ma_quantities import MaQuantities


def main():
    """input_params.toml を読み、磁気異方性エネルギーのk点数への収束を確認する。"""
    with open("input_params.toml", "r") as f:
        params = toml.load(f)
    ham_z = MagRotation(
        tb_dat=Path(params["tb_dat"]),
        nnkp_file=Path(params["nnkp_file"]),
        win_file=Path(params["win_file"]),
        extract_only_x_component=int(params["extract_only_x_component"]),
        use_convert_ham_r=int(params["use_convert_ham_r"]),
    )
    ham_x = MagRotation(
        tb_dat=Path(params["tb_dat"]),
        nnkp_file=Path(params["nnkp_file"]),
        win_file=Path(params["win_file"]),
        theta=np.pi / 2,
        extract_only_x_component=int(params["extract_only_x_component"]),
        use_convert_ham_r=int(params["use_convert_ham_r"]),
    )
    num_valence = int(params["num_valence"])
    Nk = 10
    iter = 0
    diff = 1
    dE_old = -1

    with open(Path(params["mp_folder"]) / "mae_conv.txt", "w") as f:
        f.write("Nk\t" + "dE[100] \t" + "diff\n")

    while iter < 100:
        with open(Path(params["mp_folder"]) / "mae_conv.txt", "a") as f:
            f.write(str(Nk) + "\t")
        mae_z = MaQuantities(
            ham_z, num_valence, kmesh=[Nk, Nk, Nk], win_file=Path(params["win_file"]),
        )
        mae_x = MaQuantities(
            ham_x, num_valence, kmesh=[Nk, Nk, Nk], win_file=Path(params["win_file"]),
        )
        dE_new = mae_x.free_energy - mae_z.free_energy
        with open(Path(params["mp_folder"]) / "mae_conv.txt", "a") as f:
            f.write(str(dE_new) + "\t")
            diff = np.abs(dE_new - dE_old)
            f.write(str(diff) + "\n")
        iter += 1
        Nk += 10
        dE_old = dE_new.copy()
    else:
        with open(Path(params["mp_folder"]) / "mae_conv.txt", "a") as f:
            f.write("dE converged.\n")


if __name__ == "__main__":
    main()
