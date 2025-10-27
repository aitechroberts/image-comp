# Project 1 - Image Compression

CMU 24658/42640: Image-Based Computational Modeling and Analysis

## Setup

This project uses `uv` as the package manager.

### Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies
```bash
uv sync
```

### Activate the environment
```bash
source .venv/bin/activate
```

## Problems

### Problem 1: Calculate Dog Brain Volume (10 points)

**Prerequisites:** You must first segment the brain using VolumeRover:
1. Open `mri_dcm.rawiv` in VolumeRover_seg.exe
2. Use the segmentation tool to extract the dog brain
3. This creates `mri_dcm.rawiv_subunit_00.rawiv` (brain) and `mri_dcm.rawiv_subunit_01.rawiv` (background)

**Run the script:**
```bash
# With default filename (mri_dcm.rawiv_subunit_00.rawiv)
uv run python problem1_volume_calculator.py

# Or specify a different segmented file
uv run python problem1_volume_calculator.py path/to/file.rawiv
```

This will:
- Automatically detect data type (8-bit or 16-bit)
- Read the segmented brain file efficiently using numpy
- Calculate the volume in mm³
- Print detailed statistics (dimensions, voxel spacing, brain voxel count, total volume)

### Problem 2: Isosurface Potential Mapping (20 points)

**Prerequisites:** You must first generate the isosurface using VolumeRover:
1. Open `2BG9_acc97129.rawiv` in VolumeRover
2. Set an isocontour in the colormap editor
3. Export the isosurface as a `*_tri.raw` file

**Run the script:**
```bash
uv run python problem2_potential_mapper.py
```

This will:
- Read the isosurface from the .raw file
- Calculate electron static potential for each vertex using trilinear interpolation
- Apply color mapping (red/white/blue based on potential values)
- Write a `*_tri.rawc` file
- Print min, max, and mean potential statistics

**Visualization:**
Open the `*_tri.rawc` file in MeshViewer.exe to see the colored isosurface.

## File Formats

### .rawiv (Big Endian!)
- 68-byte header containing:
  - min/max coordinates (6 floats)
  - numVerts, numCells (2 unsigned ints)
  - dimX, dimY, dimZ (3 unsigned ints)
  - originX, originY, originZ (3 floats)
  - spanX, spanY, spanZ (3 floats)
- Binary data follows, stored as floats or unsigned shorts
- **Important:** Data is in Big Endian format (even on Intel systems!)

### .raw (ASCII)
- Header: `num_vertices num_triangles`
- Vertices: one per line with x, y, z coordinates
- Triangles: one per line with 3 vertex indices

### .rawc (ASCII)
- Header: `num_vertices num_triangles`
- Vertices: one per line with x, y, z, r, g, b (colors in [0, 1])
- Triangles: one per line with 3 vertex indices

