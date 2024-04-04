#!/usr/bin/env python

"""
Description:
    Create input file for exchange.py.

Usage:
    exchange_input.py (<exc_in_toml>) [<work_dir>]
    exchange_input.py [-h | --help]

Options:
    -h --help   Show this help screen.

Author:
    Takashi Koretsune and Katsuhiro Arimoto
"""

from typing import List
from pathlib import Path
from warnings import warn

from tomli import load
from tomli_w import dump

def main():
    from docopt import docopt

    args = docopt(__doc__)
    with open(Path(args["<exc_in_toml>"]), mode="rb") as fp:
        toml_in = load(fp)
    work_dir = Path(args["<work_dir>"]) if args["<work_dir>"] else Path()
    nscf_out = toml_in.get("nscf_out", "nscf.out")
    chem_pot = get_chem_pot(work_dir/nscf_out)
    w90_prefix = toml_in.get("w90_prefix", "pwscf")
    mp_grid = get_mp_grid(work_dir/f"{w90_prefix}.wout")

    toml_out = {
        "work_dir": str(work_dir) if work_dir.is_absolute() else str(".."/work_dir), 
        "w90_prefix": w90_prefix, 
        "chem_pot": chem_pot, 
        "temperature_K": toml_in.get("temperature_K", 300), 
        "wmax": toml_in.get("wmax", 1000), 
        "s_axis": toml_in.get("s_axis", [0, 0, 1]), 
        "nproc": toml_in.get("nproc", 1), 
        "rthres_mag_splits": toml_in.get("rthres_mag_splits", 0.5), 
        "rthres_exc_r0ii": toml_in.get("rthres_exc_r0ii", 0.1), 
    }
    exc_dir = work_dir/toml_in.get("exc_dir_name", "Exchange")
    exc_dir.mkdir(exist_ok=True)
    for nkf in toml_in.get("nk_factors", [1]):
        if nkf < 1:
            warn("k mesh should be fine than MP grid of Wannierization.")
        toml_out["kmesh"] = [int(nkf*nk) for nk in mp_grid]
        toml_out["sc_grid"] = toml_out["kmesh"]
        exc_in_name = f"exc_nkf{nkf}_in.toml"
        with open(exc_dir/exc_in_name, mode="wb") as fp:
            dump(toml_out, fp)


def get_chem_pot(nscf_out: Path) -> float:
    """
    Return Fermi level from nscf.out.

    Args:
        nscf_out (Path): nscf.out file path.

    Returns:
        float: Fermi level in unit of eV.
    """
    with open(nscf_out, mode="r") as fp:
        lines = fp.readlines()
    for l in lines:
        if "the Fermi energy" in l:
            return float(l.split()[4])
    raise ValueError(f"Fermi energy is not found in {nscf_out.parent}.")


def get_mp_grid(wout: Path) -> List[int]:
    """
    Return Monkhorst-Pack k-mesh grid from wannier90.wout.

    Args:
        wout (Path): wannier90.wout file path.

    Returns:
        List[int]: k-mesh.
    """
    with open(wout, mode="r") as fp:
        lines = fp.readlines()
    for l in lines:
        if "Grid size" in l:
            return [int(n) for n in l.split()[3:8:2]]
    raise ValueError(f"MP grid is not found in {str(wout)}.")


if __name__ == "__main__":
    main()
