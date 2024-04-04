# -*- coding: utf-8 -*-

"""Ma_interface で計算した result.toml を全物質で読み取り、結果をプロットする。

Example:
    >>> python src/ma_plot.py

Todo:
    * ICSDに載っているか？はどう判定するか。
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
import matplotlib.pylab as plt
import toml
from ma_interface import Ma_interface, Inputs_path


class Inputs_for_plot:
    """プロットに必要なインプットを格納するクラス。"""

    def __init__(
        self,
        dE: float,
        dE_opt: float,
        mag_diff_ave: float,
        inputs_path: Inputs_path,
    ):
        self.dE = dE
        self.dE_opt = dE_opt
        self.mag_diff_ave = mag_diff_ave
        self.inputs_path = inputs_path


class Ma_plot:
    def __init__(self, root_dirs: List[Path]):
        """ns1 ~ ns11 の全 mp-* の形のフォルダ中のうち、
        result.toml が存在するもののパスを取得し、
        誤差が十分小さいものをリストで返すコンストラクタ。
        """
        _inputs_pathes_all = []
        self.inputs_for_plots = []
        for root_dir in root_dirs:
            ma = Ma_interface(root_dir=Path(root_dir))
            _inputs_pathes_all.extend(ma.inputs_pathes)
            _inputs_for_plot = self.check_result_exists(ma)
            self.inputs_for_plots.extend(_inputs_for_plot)
            print(
                "result.toml exists: {}/{}".format(
                    len(_inputs_for_plot), len(ma.inputs_pathes)
                )
            )
        print(
            "\nall mps calculated result.toml: {}/{}".format(
                len(self.inputs_for_plots), len(_inputs_pathes_all)
            )
        )

        self.inputs_for_plots = self.check_calc_err_small(self.inputs_for_plots)
        self.inputs_for_plots = self.sort_dE_opt(self.inputs_for_plots)
        self.plot_all_results(self.inputs_for_plots)

    def check_result_exists(self, ma: Ma_interface) -> List[Inputs_for_plot]:
        """ある root_dir 以下から results.toml までの計算が終わってるものだけを取り出し、
        dE, M_diff_ave, dE_opt を読みだし、インスタンスに格納する。"""
        inputs_for_plots = []
        for i in range(len(ma.inputs_pathes)):
            try:
                with open(
                    ma.inputs_pathes[i].mp_folder / "input_params.toml", "r"
                ) as f:
                    params = toml.load(f)
                with open(
                    ma.inputs_pathes[i].mp_folder / params["result_file_name"], "r"
                ) as f:
                    result = toml.load(f)

                _input_for_plot = Inputs_for_plot(
                    float(result["dE[100] [meV]"]),
                    float(result["dE_opt [meV]"]),
                    float(result["M_diff_ave [meV]"]),
                    ma.inputs_pathes[i],
                )
                inputs_for_plots.append(_input_for_plot)

            except:
                print(
                    "result.toml is not found in {}.".format(
                        ma.inputs_pathes[i].mp_folder.name
                    )
                )
                continue
        return inputs_for_plots

    def check_calc_err_small(
        self, inputs_for_plots: List[Inputs_for_plot], factor=10
    ) -> List[Inputs_for_plot]:
        """dE_opt が M_diff_ave の factor 倍以上のmpフォルダのみを抽出する。"""
        inputs_for_plots_checked = []
        for i in range(len(inputs_for_plots)):
            if (
                self.inputs_for_plots[i].mag_diff_ave * factor
                < self.inputs_for_plots[i].dE_opt
            ):
                _input_for_plot = Inputs_for_plot(
                    self.inputs_for_plots[i].dE,
                    self.inputs_for_plots[i].dE_opt,
                    self.inputs_for_plots[i].mag_diff_ave,
                    self.inputs_for_plots[i].inputs_path,
                )
                inputs_for_plots_checked.append(_input_for_plot)
        print(
            "Small calc error: {}/{}".format(
                len(inputs_for_plots_checked), len(inputs_for_plots)
            )
        )
        return inputs_for_plots_checked

    def sort_dE_opt(
        self, inputs_for_plots: List[Inputs_for_plot]
    ) -> List[Inputs_for_plot]:
        """dE の計算結果と、mag_diff_ave の計算結果を、dE_opt の大きい順にソートする。"""
        _dE_opt = np.array([_.dE_opt for _ in inputs_for_plots])
        idx = np.argsort(np.abs(_dE_opt))[::-1]
        inputs_for_plots_sorted = np.array(inputs_for_plots)
        inputs_for_plots_sorted = inputs_for_plots_sorted[idx]
        print("Sorted!")
        return inputs_for_plots_sorted

    def plot_all_results(self, inputs_for_plots: List[Inputs_for_plot]):
        """dE の計算結果と、mag_diff_ave の計算結果を、dE_opt の大きい順にプロットする。"""
        x = np.arange(len(inputs_for_plots))
        name = np.zeros(len(inputs_for_plots), dtype=object)
        for i in range(len(inputs_for_plots)):
            name[i] = "{} \n({})".format(
                inputs_for_plots[i].inputs_path.formula_unit,
                inputs_for_plots[i].inputs_path.mp_folder.name,
            )
        _dE_opt = np.array([_.dE_opt for _ in inputs_for_plots])
        plt.rcParams["figure.subplot.bottom"] = 0.2
        plt.figure()
        plt.bar(x[:20], _dE_opt[:20], label="$dE_{opt}$")
        # plt.bar(x[:20], np.abs(_mag_diff_ave)[:20], label="$dM_{ave}$")
        plt.xticks(x[:20], name[:20], rotation=-90, fontsize=8)
        plt.ylabel("$\Delta E_{opt} ~[\mathrm{meV}/f.u.]$")
        plt.grid()
        plt.savefig("./results/mae_results_all.pdf")


if __name__ == "__main__":
    pwd = os.getcwd()

    ns_folders = [
        Path(pwd + "/../ns1"),
        Path(pwd + "/../ns2"),
        Path(pwd + "/../ns3"),
        Path(pwd + "/../ns4"),
        Path(pwd + "/../ns5"),
        Path(pwd + "/../ns6"),
        Path(pwd + "/../ns7"),
        Path(pwd + "/../ns8"),
        Path(pwd + "/../ns9"),
        Path(pwd + "/../ns10"),
        Path(pwd + "/../ns11"),
    ]
    # ns_folders = [Path(pwd + "/../ns2")]

    ma_plot = Ma_plot(ns_folders)
