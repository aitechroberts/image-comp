#!/usr/bin/env python3
"""
Problem 2: Map electron static potential values to isosurface.

This script:
1. Reads a .raw isosurface file (ASCII format)
2. Reads potential data from a .rawiv file
3. Calculates potential values for each vertex using trilinear interpolation
4. Applies color mapping and writes a .rawc file
5. Prints statistics (min, max, mean)
"""

import struct
import sys
from typing import Dict, List, Tuple
import numpy as np


def read_raw_header(filename: str) -> Dict:
    """Read the header line from a .raw file."""
    with open(filename, 'r') as f:
        line = f.readline().strip().split()
        num_vertices = int(line[0])
        num_triangles = int(line[1])
    
    return {
        'num_vertices': num_vertices,
        'num_triangles': num_triangles
    }


def read_raw_geometry(filename: str) -> Tuple[List, List]:
    """Read vertex coordinates and triangle indices from .raw file."""
    header = read_raw_header(filename)
    
    vertices = []
    triangles = []
    
    with open(filename, 'r') as f:
        # Skip header line
        f.readline()
        
        # Read vertices
        for i in range(header['num_vertices']):
            line = f.readline().strip().split()
            x, y, z = float(line[0]), float(line[1]), float(line[2])
            vertices.append([x, y, z])
        
        # Read triangles
        for i in range(header['num_triangles']):
            line = f.readline().strip().split()
            v1, v2, v3 = int(line[0]), int(line[1]), int(line[2])
            triangles.append([v1, v2, v3])
    
    return vertices, triangles


def read_rawiv_header(filename: str) -> Dict:
    """Read the 68-byte header from a .rawiv file (Big Endian)."""
    with open(filename, 'rb') as f:
        minX, minY, minZ = struct.unpack('>fff', f.read(12))
        maxX, maxY, maxZ = struct.unpack('>fff', f.read(12))
        numVerts, numCells = struct.unpack('>II', f.read(8))
        dimX, dimY, dimZ = struct.unpack('>III', f.read(12))
        originX, originY, originZ = struct.unpack('>fff', f.read(12))
        spanX, spanY, spanZ = struct.unpack('>fff', f.read(12))
    
    return {
        'min': (minX, minY, minZ),
        'max': (maxX, maxY, maxZ),
        'dimensions': (dimX, dimY, dimZ),
        'origin': (originX, originY, originZ),
        'spans': (spanX, spanY, spanZ)
    }


def read_rawiv_data(filename: str, header: Dict) -> np.ndarray:
    """Read voxel data from .rawiv file and return as numpy array."""
    dimX, dimY, dimZ = header['dimensions']
    
    with open(filename, 'rb') as f:
        f.seek(68)  # Skip header
        
        data = []
        for z in range(dimZ):
            for y in range(dimY):
                for x in range(dimX):
                    value = struct.unpack('>f', f.read(4))[0]  # Float values
                    data.append(value)
    
    # Reshape to 3D array
    data = np.array(data).reshape(dimZ, dimY, dimX)
    return data


def trilinear_interpolation(data: np.ndarray, pos: Tuple[float, float, float], 
                            header: Dict) -> float:
    """
    Perform trilinear interpolation at position pos.
    
    Args:
        data: 3D numpy array of potential values
        pos: (x, y, z) position in world coordinates
        header: rawiv header dictionary
    
    Returns:
        Interpolated potential value
    """
    x, y, z = pos
    
    # Get grid parameters
    minX, minY, minZ = header['min']
    spanX, spanY, spanZ = header['spans']
    dimX, dimY, dimZ = header['dimensions']
    
    # Convert to grid coordinates (might be outside [0, dim-1])
    i = (x - minX) / spanX
    j = (y - minY) / spanY
    k = (z - minZ) / spanZ
    
    # Clamp to valid range
    i = max(0, min(dimX - 1.001, i))
    j = max(0, min(dimY - 1.001, j))
    k = max(0, min(dimZ - 1.001, k))
    
    # Get integer and fractional parts
    i0, i1 = int(i), int(i) + 1
    j0, j1 = int(j), int(j) + 1
    k0, k1 = int(k), int(k) + 1
    
    u = i - int(i)
    v = j - int(j)
    w = k - int(k)
    
    # Clamp indices to array bounds
    i0 = min(i0, dimX - 1)
    i1 = min(i1, dimX - 1)
    j0 = min(j0, dimY - 1)
    j1 = min(j1, dimY - 1)
    k0 = min(k0, dimZ - 1)
    k1 = min(k1, dimZ - 1)
    
    # Get the 8 corner values
    c000 = data[k0, j0, i0]
    c001 = data[k0, j0, i1]
    c010 = data[k0, j1, i0]
    c011 = data[k0, j1, i1]
    c100 = data[k1, j0, i0]
    c101 = data[k1, j0, i1]
    c110 = data[k1, j1, i0]
    c111 = data[k1, j1, i1]
    
    # Trilinear interpolation
    c00 = c000 * (1 - u) + c001 * u
    c01 = c010 * (1 - u) + c011 * u
    c10 = c100 * (1 - u) + c101 * u
    c11 = c110 * (1 - u) + c111 * u
    
    c0 = c00 * (1 - v) + c01 * v
    c1 = c10 * (1 - v) + c11 * v
    
    c = c0 * (1 - w) + c1 * w
    
    return c


def map_potential_to_color(potential: float) -> Tuple[float, float, float]:
    """
    Map potential value to RGB color according to the spec:
    - potential > 0.1  -> blue (0, 0, 1)
    - potential < -0.1 -> red (1, 0, 0)
    - potential in [-0.1, 0.1] -> white (1, 1, 1)
    """
    if potential > 0.1:
        return (0.0, 0.0, 1.0)  # Blue
    elif potential < -0.1:
        return (1.0, 0.0, 0.0)  # Red
    else:
        return (1.0, 1.0, 1.0)  # White


def write_rawc_file(filename: str, vertices: List, triangles: List, 
                    colors: List):
    """Write vertices with colors and triangles to .rawc file."""
    with open(filename, 'w') as f:
        # Write header
        f.write(f"{len(vertices)} {len(triangles)}\n")
        
        # Write vertices with colors
        for i, (v, c) in enumerate(zip(vertices, colors)):
            f.write(f"{v[0]} {v[1]} {v[2]} {c[0]} {c[1]} {c[2]}\n")
        
        # Write triangles
        for tri in triangles:
            f.write(f"{tri[0]} {tri[1]} {tri[2]}\n")
    
    print(f"Written {filename}")


def main():
    # Check for required files
    import os
    
    isosurface_file = input("Enter the .raw isosurface filename (e.g., 2BG9_tri.raw): ").strip()
    if not os.path.exists(isosurface_file):
        print(f"Error: {isosurface_file} not found!")
        print("\nYou need to first generate the isosurface using VolumeRover:")
        print("1. Open 2BG9_acc97129.rawiv in VolumeRover")
        print("2. Set isocontour in the colormap editor")
        print("3. Export the isosurface as a .raw file")
        sys.exit(1)
    
    potential_file = '2BG9_pot97129.rawiv'
    if not os.path.exists(potential_file):
        print(f"Error: {potential_file} not found!")
        sys.exit(1)
    
    print("Reading isosurface geometry...")
    vertices, triangles = read_raw_geometry(isosurface_file)
    print(f"Read {len(vertices)} vertices and {len(triangles)} triangles")
    
    print("\nReading potential data...")
    pot_header = read_rawiv_header(potential_file)
    pot_data = read_rawiv_data(potential_file, pot_header)
    print(f"Potential data dimensions: {pot_header['dimensions']}")
    
    print("\nCalculating potential values for each vertex...")
    potentials = []
    for i, vertex in enumerate(vertices):
        potential = trilinear_interpolation(pot_data, tuple(vertex), pot_header)
        potentials.append(potential)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(vertices)} vertices...")
    
    # Calculate statistics
    pot_array = np.array(potentials)
    print(f"\nPotential Statistics:")
    print(f"  Minimum: {np.min(pot_array):.6f}")
    print(f"  Maximum: {np.max(pot_array):.6f}")
    print(f"  Mean: {np.mean(pot_array):.6f}")
    
    # Map potentials to colors
    print("\nMapping potentials to colors...")
    colors = [map_potential_to_color(p) for p in potentials]
    
    # Write .rawc file
    output_file = isosurface_file.replace('.raw', '.rawc')
    write_rawc_file(output_file, vertices, triangles, colors)
    
    print(f"\nDone! You can now visualize {output_file} in MeshViewer.exe")


if __name__ == '__main__':
    main()

