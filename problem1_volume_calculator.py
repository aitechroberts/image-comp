#!/usr/bin/env python3
"""
Problem 1: Calculate the volume of the segmented dog brain.

This script reads a .rawiv file (Big Endian format) and calculates
the volume of the segmented brain tissue in mm³.
"""

import struct
import sys
import os
import numpy as np


def read_rawiv_header(fpath):
    """Read the 68-byte header from a .rawiv file (Big Endian)."""
    with open(fpath, "rb") as f:
        mins = struct.unpack(">fff", f.read(12))
        maxs = struct.unpack(">fff", f.read(12))
        nverts, ncells = struct.unpack(">II", f.read(8))
        dimx, dimy, dimz = struct.unpack(">III", f.read(12))
        origin = struct.unpack(">fff", f.read(12))
        spans = struct.unpack(">fff", f.read(12))
    return {
        "mins": mins, "maxs": maxs,
        "nverts": nverts, "ncells": ncells,
        "dims": (dimx, dimy, dimz),
        "origin": origin, "spans": spans
    }


def infer_dtype(fpath, nvox):
    """Infer data type from file size."""
    datasize = os.path.getsize(fpath) - 68
    if datasize == nvox:
        return np.dtype(">u1")  # 8-bit
    if datasize == 2 * nvox:
        return np.dtype(">u2")  # 16-bit
    raise ValueError(f"Unexpected data size: {datasize} bytes for {nvox} voxels")


def load_volume(fpath, dims, dtype):
    """Load volume data using numpy for fast bulk reading."""
    with open(fpath, "rb") as f:
        f.seek(68)
        arr = np.fromfile(f, dtype=dtype, count=dims[0] * dims[1] * dims[2])
    return arr.reshape(dims[::-1])  # (Z,Y,X) if you care about indexing


def calc_volume_mm3(fpath):
    """Calculate volume and return statistics."""
    hdr = read_rawiv_header(fpath)
    dims = hdr["dims"]
    spans = hdr["spans"]  # (mm, mm, mm)
    nvox = dims[0] * dims[1] * dims[2]
    dtype = infer_dtype(fpath, nvox)
    vol = load_volume(fpath, dims, dtype)
    voxel_mm3 = spans[0] * spans[1] * spans[2]
    brain_voxels = int(np.count_nonzero(vol))
    return brain_voxels * voxel_mm3, brain_voxels, voxel_mm3, dims, spans, dtype


def main():
    # Accept filename argument, default to segmented file
    fpath = sys.argv[1] if len(sys.argv) > 1 else "mri_dcm.rawiv_subunit_00.rawiv"
    
    if not os.path.exists(fpath):
        print(f"Segmented file not found: {fpath}")
        print("\nYou need to first segment the brain using VolumeRover_seg.exe:")
        print("1. Open mri_dcm.rawiv in VolumeRover")
        print("2. Use the segmentation tool to extract the dog brain")
        print("3. This will create mri_dcm.rawiv_subunit_00.rawiv (brain)")
        print("   and mri_dcm.rawiv_subunit_01.rawiv (background)")
        sys.exit(1)
    
    total_mm3, n, vmm3, dims, spans, dtype = calc_volume_mm3(fpath)
    print(f"File: {fpath}")
    print(f"Dimensions: {dims[0]} x {dims[1]} x {dims[2]}")
    print(f"Voxel spacing (mm): {spans[0]:.6f} x {spans[1]:.6f} x {spans[2]:.6f}  -> voxel volume {vmm3:.6f} mm^3")
    print(f"Data type: {dtype}")
    print(f"Non-zero voxels (brain): {n:,}")
    print(f"Brain volume: {total_mm3:.2f} mm^3")


if __name__ == '__main__':
    main()

