# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import os
import subprocess

tests_path = os.path.abspath(os.path.dirname(__file__))
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(src_path)
import numpy as np
from mag_rotation import MagRotation
from ma_quantities import MaQuantities
from ma_interface import MaInterface

num_valence = 18
hr_dat = Path(tests_path) / "mp-2260/pwscf_rel_sym/mae/wan/pwscf_py_hr.dat"
tb_dat = Path(tests_path) / "mp-2260/pwscf_rel_sym/mae/wan/pwscf_py_tb.dat"
win_file = Path(tests_path) / "mp-2260/pwscf_rel_sym/mae/pwscf.win"
nnkp_file = Path(tests_path) / "mp-2260/pwscf_rel_sym/mae/wan/pwscf.nnkp"
work_dir = Path(tests_path) / "mp-2260/"


def test_E_F():
    # フェルミエネルギーが単一コアと並列化で一致するか？
    ham_rotated = MagRotation(tb_dat=tb_dat, extract_only_x_component=True,)
    Nk = 20
    mae_serial = MaQuantities(
        ham_rotated, num_valence, kmesh=[Nk, Nk, Nk], win_file=win_file
    )
    mae_parallel = MaQuantities(
        ham_rotated,
        num_valence,
        kmesh=[Nk, Nk, Nk],
        win_file=win_file,
        calc_spin_angular_momentum=True,
    )
    assert mae_serial.sorted_eigvec == None  # Dose the parallelization work?
    assert np.abs(mae_serial.fermi_energy - mae_parallel.fermi_energy) < 1e-9


"""
def test_spin_moment():
    # Is the spin angular momentum correct?
    ham_rotated = MagRotation(
        tb_dat=tb_dat,
        nnkp_file=nnkp_file,
        win_file=win_file,
    )
    mae = MaQuantities(
        ham_rotated,
        num_valence,
        win_file=win_file,
        calc_spin_angular_momentum=True,
    )
    assert np.abs(mae.spin_angular_momentum[0] - -3.309276760889407) < 1e-9
    assert np.abs(mae.spin_angular_momentum[1] - -2.2888910608988565e-6) < 1e-9
    assert np.abs(mae.spin_angular_momentum[2] - -2.2947263418540696e-6) < 1e-9


def test_work_dir():
    # 一時ファイルの読み書きを用いたスピン磁化の計算は正常に行えているか？
    ham_rotated = MagRotation(tb_dat=tb_dat)
    mae = MaQuantities(
        ham_rotated,
        num_valence,
        win_file=win_file,
        calc_spin_angular_momentum=True,
        work_dir=work_dir,
    )
    assert np.abs(mae.spin_angular_momentum[0] - -3.309276760889407) < 1e-9
    assert np.abs(mae.spin_angular_momentum[1] - -2.2888910608988565e-6) < 1e-9
    assert np.abs(mae.spin_angular_momentum[2] - -2.2947263418540696e-6) < 1e-9
"""

"""
def test_orbital_moment():
    # Is the orbital angular momentum correct?
    ham_rotated = MagRotation(
        tb_dat=tb_dat, 
        extract_only_x_component=True,
        nnkp_file=nnkp_file, 
        win_file=win_file
    )
    mae = MaQuantities(
        ham_rotated,
        num_valence,
        work_dir=work_dir,
        win_file=win_file,
        write_and_read_work=True,
        calc_orbital_angular_momentum=True,
    )
    assert np.abs(mae.orbital_angular_momentum[0] - 3.887498584988353e-7) < 1e-9
    assert np.abs(mae.orbital_angular_momentum[1] - -1.0044356515634367e-6) < 1e-9
    assert np.abs(mae.orbital_angular_momentum[2] - -0.2961517255339583) < 1e-9
"""


def test_write_hr():
    # 磁化回転後の hr.dat が正しく書き出せているか？
    ham_orig = MagRotation(
        hr_dat=hr_dat, extract_only_x_component=False, write_rotated_hr=True,
    )
    ham_rot = MagRotation(
        hr_dat=hr_dat.parent / "pwscf_rot_hr.dat", extract_only_x_component=False,
    )
    assert np.all(np.abs(ham_orig.hrs - ham_rot.hrs) < 1e-10)


def test_interface():
    # インターフェイスの input.toml の生成が test フォルダ内で機能しているか？
    ma = MaInterface(root_dir=Path(tests_path))
    try:
        ma.make_input_toml()
        assert True
    except:
        assert False


"""
def test_parallel_eigval():
    # qsubでmpi並列が行えているか？
    try:
        with open(Path(tests_path) / "submit.sh", "w") as f:
            script_content = (
                "#!/bin/bash \n"
                + "#PBS -l nodes=1:ppn=8 \n"
                + "#PBS -q GroupE \n\n"
                + "source /opt/intel_2022/setvars.sh --force intel64 \n"
                + "conda activate mae \n"
                + "cd $PBS_O_WORKDIR \n"
            )
            python_script = (
                "numactl --interleave=all "
                + "mpirun -n 8 python {}\n".format(
                    Path(src_path) / "parallel_eigval.py"
                )
                + "rm -f submit.sh.*"
            )
            f.write(script_content + python_script)
        subprocess.run(
            "cd {}; qsub submit.sh".format(Path(tests_path)),
            shell=True,
            check=True,
        )
        assert True
    except:
        assert False
"""
