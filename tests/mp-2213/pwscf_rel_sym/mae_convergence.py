from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import toml
import sys

sys.path.append("../../../src")

from mag_rotation import MagRotation
from ma_quantities import MaQuantities


class Mae_convergence:
    def __init__(self):
        """input_params.toml を読み、磁気異方性エネルギーのk点数への収束を確認する。"""
        ham_z = MagRotation(
            tb_dat=Path("mae/wan/pwscf_py_tb.dat"),
            extract_only_x_component=0
        )
        ham_x = MagRotation(
            tb_dat=Path("mae1/wan/pwscf_py_tb.dat"),
            extract_only_x_component=0,
        )
        num_valence = 18
        Nk = 10
        iter = 0
        diff = 1
        dE_old = -1

        with open("./mae_conv.txt", "w") as f:
            f.write("Nk\t" + "dE[100] \t" + "diff\n")

        while (diff > 1e-16) and (iter < 100):
            with open(
                "./mae_conv.txt", "a"
            ) as f:
                f.write(str(Nk) + "\t")
            mae_z = MaQuantities(
                ham_z,
                num_valence,
                kmesh=[Nk, Nk, Nk],
                win_file=Path("mae1/pwscf.win"),
            )
            mae_x = MaQuantities(
                ham_x,
                num_valence,
                kmesh=[Nk, Nk, Nk],
                win_file=Path("mae1/pwscf.win"),
            )
            dE_new = mae_x.free_energy - mae_z.free_energy
            with open(
                "./mae_conv.txt", "a"
            ) as f:
                f.write(str(dE_new) + "\t")
                diff = np.abs(dE_new - dE_old)
                f.write(str(diff) + "\n")
            iter += 1
            Nk += 10
            dE_old = dE_new.copy()
        else:
            with open(
                "./mae_conv.txt", "a"
            ) as f:
                f.write("dE converged.\n")


if __name__ == "__main__":
    Mae_convergence()
