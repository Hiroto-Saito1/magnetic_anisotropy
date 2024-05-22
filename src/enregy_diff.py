# -*- coding: utf-8 -*-

"""
TB-EM法, TB-OM法とDFTのそれぞれで磁化を回転させた場合の、すべてのエネルギー固有値の絶対値の平均を計算する。

Example:
    >>> cd path/to/input_params.toml
    >>> mpirun -n 8 python ma_quantities.py
"""
from pathlib import Path
import toml
import numpy as np


from mag_rotation import MagRotation
from ma_quantities import MaQuantities


class EnergyDiff:
    """
    TB-EM法, TB-OM法とDFTのそれぞれで磁化を回転させた場合の、すべてのエネルギー固有値のDFTとの差の絶対値の平均を計算する。
    """

    def __init__(self):
        self.read_input_toml()
        ham_TB_EM, ham_TB_OM, ham_DFT = self.make_hamiltonians_along_x()
        TB_EM, TB_OM, DFT = self.calc_sorted_energy(ham_TB_EM, ham_TB_OM, ham_DFT)
        self.write_output_toml(TB_EM, TB_OM, DFT)

    def read_input_toml(self):
        with open("input_params.toml", "r") as f:
            params = toml.load(f)
        self.tb_dat_z = Path(params["tb_dat_z"])
        self.tb_dat_x = Path(params["tb_dat_x"])
        self.win_file = Path(params["win_file"])
        self.nnkp_file = Path(params["nnkp_file"])
        self.mp_folder = Path(params["mp_folder"])
        self.Nk = int(params["Nk"])
        self.kmesh = [self.Nk, self.Nk, self.Nk]
        self.num_valence = int(params["num_valence"])

    def make_hamiltonians_along_x(self):
        """
        3種類の方法で、磁化をx軸方向に向けたハミルトニアンを用意する。
        """
        ham_TB_EM = MagRotation(
            tb_dat=self.tb_dat_z,
            extract_only_x_component=True,
            use_convert_ham_r=False,
            nnkp_file=self.nnkp_file,
            win_file=self.win_file,
            theta=np.pi / 2,
        )

        ham_TB_OM = MagRotation(
            tb_dat=self.tb_dat_z,
            extract_only_x_component=False,
            use_convert_ham_r=False,
            nnkp_file=self.nnkp_file,
            win_file=self.win_file,
            theta=np.pi / 2,
        )

        ham_DFT = MagRotation(
            tb_dat=self.tb_dat_x,
            extract_only_x_component=False,
            use_convert_ham_r=False,
            nnkp_file=self.nnkp_file,
            win_file=self.win_file,
        )
        return ham_TB_EM, ham_TB_OM, ham_DFT

    def calc_sorted_energy(
        self, ham_TB_EM: MagRotation, ham_TB_OM: MagRotation, ham_DFT: MagRotation
    ):
        TB_EM = MaQuantities(
            ham_TB_EM,
            self.num_valence,
            kmesh=self.kmesh,
            win_file=self.win_file,
        )

        TB_OM = MaQuantities(
            ham_TB_OM,
            self.num_valence,
            kmesh=self.kmesh,
            win_file=self.win_file,
        )

        DFT = MaQuantities(
            ham_DFT,
            self.num_valence,
            kmesh=self.kmesh,
            win_file=self.win_file,
        )
        return TB_EM, TB_OM, DFT

    def write_output_toml(
        self, TB_EM: MaQuantities, TB_OM: MaQuantities, DFT: MaQuantities
    ):
        diff_TB_EM = np.mean(np.abs(TB_EM.sorted_energy - DFT.sorted_energy))
        diff_TB_OM = np.mean(np.abs(TB_OM.sorted_energy - DFT.sorted_energy))

        if np.any(TB_EM.fermi_energy != None):
            result_contents = {
                "Nk": self.Nk,
                "diff TB_EM [meV]": float(diff_TB_EM * 10**3),
                "diff TB_OM [meV]": float(diff_TB_OM * 10**3),
            }

        with open(self.mp_folder / "result.toml", "w") as f:
            toml.dump(result_contents, f)


if __name__ == "__main__":
    EnergyDiff()
