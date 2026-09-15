#!/usr/bin/env python3
"""Create a dimensioned cuboid CAD mesh for an existing pose model."""
import argparse
from pathlib import Path

from object_catalog import DIMENSIONS as SIZES
FACES = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6),
         (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
         (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1)]


def write_mesh(path: Path, dims: tuple[float, float, float]):
    x, y, z = [v/2 for v in dims]
    vertices = [(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),
                (-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('# centered metric cuboid; local X is the long side\n')
        for v in vertices:
            f.write('v %.8f %.8f %.8f\n' % v)
        for face in FACES:
            f.write('f %d %d %d\n' % face)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=SIZES, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    write_mesh(args.out, SIZES[args.kind])
