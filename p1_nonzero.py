#!/usr/bin/env python3
"""
Problem 1 (nonzero): Calculate brain volume by counting ALL non-zero voxels.

- Reads .rawiv (Big Endian) header and data
- Builds mask as (voxel != 0) regardless of data type
- Counts voxels and multiplies by physical voxel volume (mm^3)
"""
import os
import struct
import sys
from typing import Tuple

import numpy as np


def read_rawiv_header(fpath: str):
    with open(fpath, "rb") as f:
        mins = struct.unpack(">fff", f.read(12))
        maxs = struct.unpack(">fff", f.read(12))
        nverts, ncells = struct.unpack(">II", f.read(8))
        dimx, dimy, dimz = struct.unpack(">III", f.read(12))
        origin = struct.unpack(">fff", f.read(12))
        spans = struct.unpack(">fff", f.read(12))
    return {
        "mins": mins,
        "maxs": maxs,
        "nverts": nverts,
        "ncells": ncells,
        "dims": (dimx, dimy, dimz),
        "origin": origin,
        "spans": spans,
    }


def infer_dtype(fpath: str, nvox: int) -> np.dtype:
    datasize = os.path.getsize(fpath) - 68
    if datasize == nvox:
        return np.dtype(">u1")  # 8-bit
    if datasize == 2 * nvox:
        return np.dtype(">u2")  # 16-bit
    if datasize == 4 * nvox:
        return np.dtype(">f4")  # 32-bit float
    raise ValueError(f"Unexpected data size: {datasize} bytes for {nvox} voxels")


def load_volume(fpath: str, dims: Tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    with open(fpath, "rb") as f:
        f.seek(68)
        arr = np.fromfile(f, dtype=dtype, count=dims[0] * dims[1] * dims[2])
    return arr.reshape(dims[::-1])  # (Z, Y, X)


def calc_volume_mm3(fpath: str):
    hdr = read_rawiv_header(fpath)
    dims = hdr["dims"]
    spans = hdr["spans"]  # (mm, mm, mm)
    nvox = dims[0] * dims[1] * dims[2]
    dtype = infer_dtype(fpath, nvox)
    vol = load_volume(fpath, dims, dtype)
    voxel_mm3 = spans[0] * spans[1] * spans[2]

    # Count ALL non-zero voxels
    brain_voxels = int(np.count_nonzero(vol))
    return brain_voxels * voxel_mm3, brain_voxels, voxel_mm3, dims, spans, dtype


def main():
    # Accept filename argument, default to brain subunit
    fpath = sys.argv[1] if len(sys.argv) > 1 else "mri_dcm.rawiv_subunit_01.rawiv"

    if not os.path.exists(fpath):
        print(f"Segmented file not found: {fpath}")
        sys.exit(1)

    total_mm3, n, vmm3, dims, spans, dtype = calc_volume_mm3(fpath)
    print(f"File: {fpath}")
    print(f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
    print(
        f"Voxel spacing (mm): {spans[0]:.6f} x {spans[1]:.6f} x {spans[2]:.6f}  -> voxel volume {vmm3:.6f} mm^3"
    )
    print(f"Data type: {dtype}")
    print(f"Non-zero voxels: {n:,}")
    print(f"Volume (all non-zero): {total_mm3:.2f} mm^3 ({total_mm3/1000:.2f} cm^3)")


if __name__ == "__main__":
    main()


