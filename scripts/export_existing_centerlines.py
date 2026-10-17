# scripts/.py

import argparse
import csv
from pathlib import Path

import numpy as np

from navierstokes.fdm.centerline import (
    extract_centerline_profiles,
)
from navierstokes.fdm.grid import CartesianGrid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    with np.load(
        args.input / "fields.npz",
        allow_pickle=False,
    ) as data:
        x = data["x"].copy()
        y = data["y"].copy()
        u = data["u"].copy()
        v = data["v"].copy()

    grid = CartesianGrid(
        x=x,
        y=y,
        dx=float(x[1] - x[0]),
        dy=float(y[1] - y[0]),
    )

    profiles = extract_centerline_profiles(
        grid,
        u,
        v,
    )

    for filename, headers, columns in (
        (
            "centerline_u.csv",
            ["y", "u"],
            (profiles["y"], profiles["u_vertical"]),
        ),
        (
            "centerline_v.csv",
            ["x", "v"],
            (profiles["x"], profiles["v_horizontal"]),
        ),
    ):
        with (args.input / filename).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(zip(*columns))

    print(f"Perfis exportados em: {args.input}")


if __name__ == "__main__":
    main()
