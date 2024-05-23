# -*- coding: utf-8 -*-

"""mae_convergence.py, lam_dep.py, angle_dep.py, bandfilling.py を統合したインターフェイス。

Example:
    >>> python ma_interface.py

Todo:

"""

import os
import fnmatch
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import toml
import socket


class InputPath:
    """計算に必要なインプットファイルのパスを格納するクラス。"""

    def __init__(
        self,
        mp_folder: Path = None,
        win_file: Path = None,
        tb_dat: Path = None,
        scf_out: Path = None,
        nnkp_file: Path = None,
        formula_unit: str = None,
    ):
        self.mp_folder = mp_folder
        self.win_file = win_file
        self.tb_dat = tb_dat
        self.scf_out = scf_out
        self.nnkp_file = nnkp_file
        self.formula_unit = formula_unit


class MaInterface:
    """条件を満たす全てのmpフォルダについて、qsub or bsub で計算を流すクラス。"""

    def __init__(self, root_dir: Optional[Path] = None):
        """root_dir 以下の mp-* の形のフォルダのパスと計算に必要なファイルのパスを取得し、
        その中からTRS条件を満たしているディレクトリのみを残したリストを返すコンストラクタ。
        """
        mp_folders_all = subprocess.run(
            "ls -d " + str(root_dir) + "/*/",
            shell=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        mp_folders_all = mp_folders_all.decode().strip().split("\n")
        mp_folders_all = [Path(_) for _ in mp_folders_all]
        self.mp_folders_all = mp_folders_all
        _src_dir = subprocess.run(
            "find `pwd` -name ma_quantities.py",
            shell=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        _src_dir = _src_dir.decode().strip("\n")
        _src_dir = _src_dir.strip("ma_quantities.py")
        self.src_dir = Path(_src_dir)

        self.inputs_pathes, self.dft_unfinished = self.check_dft_finished(
            mp_folders_all
        )
        # self.inputs_pathes = self.check_wannier_centers_and_spreads(self.inputs_pathes)
        # self.inputs_pathes = self.check_f_electrons(self.inputs_pathes)

    def find_file_path(self, dir_path: Path, file_name_pattern: str) -> Path:
        """dir_path 以下にある file_name_pattern の絶対パスを取得する。
        ワイルドカードも使える。複数存在する場合は最初に見つかったもののみ。
        見つからない場合はエラーを返す。
        """
        for root, dirs, files in os.walk(str(dir_path)):
            for name in files:
                if fnmatch.fnmatch(name, file_name_pattern):
                    return Path(os.path.abspath(os.path.join(root, name)))
        raise ValueError(
            "Error: {} not found in {}.".format(file_name_pattern, dir_path)
        )

    def contain_f_electrons(self, formula):
        lanthanoids = [
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        ]
        actinoids = [
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
            "Md",
            "No",
            "Lr",
        ]
        f_electrons = lanthanoids + actinoids
        for element in f_electrons:
            if element in formula:
                return True
        return False

    def read_num_valence(self, path_to_scf_out: Path, path_to_pwscf_win: Path,) -> int:
        """scf.out と pwscf.win から価電子数を計算する。"""
        scf_out = open(path_to_scf_out, "r")
        num_line = sum([1 for _ in open(path_to_scf_out, "r")])
        line = ""
        for i in range(num_line):
            if "number of electrons" not in line:
                line = scf_out.readline()
            else:
                num_elec = float(line.split()[4])
        scf_out.close()

        pwscf_win = open(path_to_pwscf_win, "r")
        num_line = sum([1 for _ in open(path_to_pwscf_win, "r")])
        line = ""
        excluded_bands = 0.0
        for i in range(num_line):
            if "exclude_bands" not in line:
                line = pwscf_win.readline()
            else:
                excluded_bands = line.split()[2]
                idx = excluded_bands.find("-")
                excluded_bands = excluded_bands[idx + 1 :]
                excluded_bands = float(excluded_bands.replace(",", ""))
        pwscf_win.close()
        return int(num_elec - excluded_bands)

    def check_dft_finished(self, mp_folders_all: List[Path]) -> List[InputPath]:
        """mp_folders_all 中の全てのmpフォルダのうち、Wannier化までの処理がうまくいっているものの List[InputPath] と、
        Wannier化が終わっていないものの List[Path] を返す。
        """
        dft_unfinished = []
        inputs_pathes = []
        for mp in mp_folders_all:
            try:
                _tb_dat = self.find_file_path(mp, "*py_tb.dat.gz")
                _tb_dat = Path(str(_tb_dat)[:-3])
                _win_file = self.find_file_path(mp, "pwscf.win")
                _scf_out = self.find_file_path(mp, "scf.out")
                _nnkp_file = self.find_file_path(mp, "pwscf.nnkp")
                _formula_unit = self.find_file_path(mp, "*sym.cif")
                _formula_unit = str(_formula_unit.name).replace("_sym.cif", "")
                inputs_path = InputPath(
                    mp, _win_file, _tb_dat, _scf_out, _nnkp_file, _formula_unit
                )
                inputs_pathes.append(inputs_path)
            except:
                print("Some input files are not found in {}.".format(mp.name))
                dft_unfinished.append(mp)
                continue
        print(
            "Num of mps finished Wannierization: {}/{} \n".format(
                len(inputs_pathes), len(mp_folders_all)
            )
        )
        return inputs_pathes, dft_unfinished

    def check_wannier_centers_and_spreads(
        self, inputs_pathes: List[InputPath]
    ) -> List[InputPath]:
        """pwscf.wout からWannier軌道の中心と広がりを読み、基準に達していないものはリストから除外する。"""
        inputs_pathes_checked = []
        cnt_no_conv, cnt_conv_large, cnt_wan_large, cnt_not_trs = 0, 0, 0, 0
        for i in range(len(inputs_pathes)):
            try:
                _pwscf_wout_path = self.find_file_path(
                    inputs_pathes[i].mp_folder, "pwscf.wout"
                )
                pwscf_wout = open(_pwscf_wout_path, "r")
                num_line = sum([1 for _ in open(_pwscf_wout_path, "r")])
                line = ""
                center = []
                spread = []
                for j in range(num_line):
                    if "Final State" not in line:
                        line = pwscf_wout.readline()
                    else:
                        line = pwscf_wout.readline()
                        for k in range(num_line):
                            if "WF centre" in line:
                                center.append(
                                    [
                                        float(l.replace(",", ""))
                                        for l in line.split()[6:9]
                                    ]
                                )
                                spread.append(float(line.split()[10]))
                                line = pwscf_wout.readline()
                pwscf_wout.close()
                center = np.array(center)
                spread = np.array(spread)

                _conv_path = self.find_file_path(inputs_pathes[i].mp_folder, "CONV")
                conv = open(_conv_path, "r")
                num_line = sum([1 for _ in open(_conv_path, "r")])
                line = ""
                ave_diff = 1.0
                for j in range(num_line):
                    if "average diff" not in line:
                        line = conv.readline()
                    else:
                        ave_diff = float(line.split()[3])
                        break

                criterion = 0
                for j in range(int(len(spread) / 2)):
                    diff_spread = np.abs(spread[2 * j + 1] - spread[2 * j])
                    diff_center = np.linalg.norm(
                        center[2 * j + 1, :] - center[2 * j, :]
                    )
                    if ave_diff > 0.015:
                        cnt_conv_large += 1
                        criterion += 1
                        break
                    elif spread[2 * j] > 15:
                        cnt_wan_large += 1
                        criterion += 10
                        break
                    elif (diff_spread > 1e-8) or (diff_center > 1e-5):
                        cnt_not_trs += 1
                        criterion += 100
                        break
                if criterion == 1:
                    print(
                        "CONV is large in {}.".format(inputs_pathes[i].mp_folder.name)
                    )
                elif criterion == 10:
                    print(
                        "Wannier spreads are large in {}.".format(
                            inputs_pathes[i].mp_folder.name
                        )
                    )
                elif criterion == 100:
                    print(
                        "TRS Wannier not satisfied in {}.".format(
                            inputs_pathes[i].mp_folder.name
                        )
                    )
                else:
                    _input_path = InputPath(
                        inputs_pathes[i].mp_folder,
                        inputs_pathes[i].win_file,
                        inputs_pathes[i].tb_dat,
                        inputs_pathes[i].scf_out,
                        inputs_pathes[i].nnkp_file,
                        inputs_pathes[i].formula_unit,
                    )
                    inputs_pathes_checked.append(_input_path)
            except:
                cnt_no_conv += 1
                print(
                    "CONV file is not found in {}.".format(
                        inputs_pathes[i].mp_folder.name
                    )
                )
                continue
        print(
            "\nfinished Wannierization: {}/{}\n".format(
                len(inputs_pathes), len(self.mp_folders_all)
            )
            + "CONV not found: {}\n".format(cnt_no_conv)
            + "CONV large: {}\n".format(cnt_conv_large)
            + "Wannier spreads large: {}\n".format(cnt_wan_large)
            + "TRS condition not satisfied: {}\n".format(cnt_not_trs)
            + "passed all conditions: {}".format(len(inputs_pathes_checked))
        )
        return inputs_pathes_checked

    def check_f_electrons(self, inputs_pathes: List[InputPath]) -> List[InputPath]:
        """f電子が含まれていないmpフォルダのみを抽出する。"""
        inputs_pathes_checked = []
        for i in range(len(inputs_pathes)):
            if not self.contain_f_electrons(inputs_pathes[i].formula_unit):
                inputs_pathes_checked.append(inputs_pathes[i])
        print(
            "Not contain f electrons: {}/{}".format(
                len(inputs_pathes_checked), len(inputs_pathes)
            )
        )
        return inputs_pathes_checked

    def make_input_toml(
        self,
        extract_only_x_component: int = 1,
        use_convert_ham_r: int = 0,
        Nk: int = 10,
        mae_conv_file_name: str = "mae_conv.txt",
        angle_dep_file_name: str = "angle_dep.txt",
    ):
        """計算条件に必要なinputをまとめたtomlファイルを各フォルダで生成する。"""
        for i in range(len(self.inputs_pathes)):
            try:
                num_valence = self.read_num_valence(
                    self.inputs_pathes[i].scf_out, self.inputs_pathes[i].win_file
                )
                inpupt_params = {
                    "src_dir": str(self.src_dir),
                    "tb_dat": str(self.inputs_pathes[i].tb_dat),
                    "win_file": str(self.inputs_pathes[i].win_file),
                    "nnkp_file": str(self.inputs_pathes[i].nnkp_file),
                    "mp_folder": str(self.inputs_pathes[i].mp_folder),
                    "extract_only_x_component": extract_only_x_component,
                    "use_convert_ham_r": use_convert_ham_r,
                    "Nk": Nk,
                    "num_valence": num_valence,
                }
                with open(
                    self.inputs_pathes[i].mp_folder / "input_params.toml", "w"
                ) as f:
                    toml.dump(inpupt_params, f)
            except:
                raise ValueError(
                    "Errors happend in " + str(self.inputs_pathes[i].mp_folder.name)
                )

    def qsub_pyprogram(
        self, pyprogram: str, group="GroupC", use_mpi: bool = True, nproc=1
    ):
        """qsubでさまざまなpythonプログラムを実行する。"""
        for i in range(len(self.inputs_pathes)):
            f = open(self.inputs_pathes[i].mp_folder / "submit.sh", "w")
            script_content = (
                # "#!/bin/bash \n" +
                "#PBS -q {} \n\n".format(group)
                + "source /opt/intel_2022/setvars.sh --force intel64 \n"
                + "conda activate mae \n"
                + "cd $PBS_O_WORKDIR \n"
            )
            if use_mpi:
                script_content = f"#PBS -l nodes=1:ppn={nproc} \n" + script_content
                python_script = (
                    "numactl --interleave=all "
                    + "mpirun -n {} python {}\n".format(
                        nproc, Path(self.src_dir) / pyprogram
                    )
                    + "rm -f submit.sh.*"
                )
            else:
                python_script = (
                    "python {} \n".format(Path(self.src_dir) / pyprogram)
                    + "rm -f submit.sh.*"
                )
            f.write(script_content + python_script)
            f.close()
            subprocess.run(
                "cd {}; chmod u+x submit.sh; qsub submit.sh".format(
                    self.inputs_pathes[i].mp_folder
                ),
                shell=True,
                check=True,
            )

    def bsub_pyprogram(
        self,
        pyprogram: str,
        group='"aries01 aries02 aries03"',
        use_mpi: bool = True,
        nproc=1,
    ):
        """bsubでさまざまなpythonプログラムを実行する。"""
        for i in range(len(self.inputs_pathes)):
            f = open(self.inputs_pathes[i].mp_folder / "submit.sh", "w")
            script_content = "#!/bin/bash \n"
            if use_mpi:
                python_script = (
                    "mpirun -n {} python {} \n".format(
                        nproc, Path(self.src_dir) / pyprogram
                    )
                    + "rm -f err"
                )
            else:
                python_script = (
                    "python {} \n".format(Path(self.src_dir) / pyprogram) + "rm -f err"
                )
            f.write(script_content + python_script)
            f.close()

            subprocess.run(
                "cd {}; chmod u+x submit.sh; bsub -n {} -m {} -e err ./submit.sh".format(
                    self.inputs_pathes[i].mp_folder, nproc, group
                ),
                shell=True,
                check=True,
            )


if __name__ == "__main__":
    tests_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests"))

    ma = MaInterface(root_dir=Path(tests_path))
    
    ma.make_input_toml(
        Nk=10,
        # extract_only_x_component=1,
        # use_convert_ham_r=1,
    )

    if socket.gethostname() == "toki":
        # ma.qsub_pyprogram("energy_diff.py", use_mpi=True, nproc=10, group="GroupA")
        # ma.qsub_pyprogram("angle_dep.py", use_mpi=True, nproc=22, group="GroupC")
        # ma.qsub_pyprogram("mae_convergence.py")
        # ma.qsub_pyprogram("bandfilling.py")
        # ma.qsub_pyprogram("lam_dep.py")
        pass
    elif socket.gethostname() == "zodiac":
        ma.bsub_pyprogram("ma_quantities.py", nproc=10)
        # ma.bsub_pyprogram("angle_dep.py")
        # ma.bsub_pyprogram("mae_convergence.py")
        # ma.bsub_pyprogram("bandfilling.py")
        # ma.bsub_pyprogram("lam_dep.py")
        pass
    else:
        print("\nhostname is not toki or zodiac.")
