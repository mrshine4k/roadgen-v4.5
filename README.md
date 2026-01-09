# Procedural Road Network Generator

A high-performance, terrain-aware procedural road generation tool written in Rust. This project generates realistic, hierarchical road networks on arbitrary heightmaps by simulating organic growth patterns, favoring flat terrain, and reinforcing heavily used paths into highways.

![Rust](https://img.shields.io/badge/rust-stable-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Terrain-Aware Pathfinding:** Uses an anisotropic A\* algorithm where movement cost is heavily influenced by slope. Roads naturally wind around mountains and stick to valleys.
- **Hierarchical Network:** Roads are not static. The system tracks usage, automatically upgrading frequently traveled paths into wide highways while leaving less used connections as local roads.
- **Organic Connectivity:** Points of Interest (POIs) are generated based on local elevation maxima and clustered to create distinct "cities" or settlements.
- **Performance Optimized:**
  - Parallel processing using `rayon` for slope calculation, POI clustering, and spline generation.
  - Spatial Grid indexing for fast neighbor lookups.
  - **Galin et al. Visibility Masks:** Optimizes pathfinding jumps to prevent checking every single pixel.
- **Smoothing:** Raw paths are simplified using the Ramer-Douglas-Peucker algorithm and interpolated using Catmull-Rom splines for smooth curves.

## Example Output

The provided configuration generates a **macro-scale regional network**.

- **Visual Style:** High-contrast black background with roads ranging from grey (local) to white (highway).
- **Behavior:** Roads will aggressively avoid steep slopes (winding through valleys) and strongly prefer to merge with existing roads, creating natural highway arteries.

## Installation

### Prerequisites

- Rust toolchain (latest stable)
- Cargo

### Build

```bash
git clone https://github.com/yourusername/road-generator.git
cd road-generator
cargo build --release
```

## Usage

1.  Create an `input` directory and place your heightmap image (e.g., `heightmap.png`) inside it.
    - _Supported formats:_ PNG, JPEG, BMP, etc.
    - _Recommendation:_ Grayscale images where white = high elevation, black = low elevation.
2.  Create a `config.toml` file in the root directory (see [Configuration](#configuration) below).
3.  Run the binary:

```bash
cargo run --release
```

The generated road network image will be saved in the `output/` directory.

## Configuration

The behavior of the generator is controlled via `config.toml`.

### Example Configuration

```toml
[input]
# Path to your source terrain image
heightmap_path = "input/heightmap.png"
# Directory where generated images are saved
output_dir = "output/"

[processing]
# Maximum width a road can grow to (in pixels/units)
max_road_width = 16
# Min/Max size of generated Points of Interest (Settlements)
poi_min_size = 18
poi_max_size = 32
# Max distance (in pixels) a POI will look for a neighbor to connect to
connection_radius = 4000.0
# How much slope penalizes movement. Higher = roads wind more to stay flat.
slope_weight = 900.0
# Factor (α) that reduces cost if a path overlaps an existing road.
# Higher = Stronger incentive to merge into highways.
road_reuse_bonus = 5.0

[pathfinding]
# How aggressively the raw path is simplified before smoothing.
simplify_epsilon = 7.0
# Max number of neighbors each POI tries to connect to.
max_connections_per_poi = 16
# Integer mask size (k) for A* jumps. Higher = faster but less granular.
mask_size = 5

[visualization]
# Background color of the output image
heightmap_color = [0, 0, 0]
# Colors for roads based on usage index (1 to 5)
road_colors = [
  [160, 160, 160],    # usage=1 (local road)
  [180, 180, 180],    # usage=2
  [200, 200, 200],    # usage=3
  [220, 220, 220],    # usage=4
  [255, 255, 255]     # usage=5 (highway)
]
# Color of the Points of Interest (Cities)
poi_color = [255, 60, 60]
# Color of spline control points (if draw_splines is true)
spline_color = [60, 255, 60]
draw_splines = false

[random]
# Seed for deterministic generation
seed = 42
# Density of POIs (Lower = fewer, larger cities)
poi_density = 0.000002
```

### Parameter Guide

| Parameter           | Effect of Increasing Value                                                                                      |
| ------------------- | --------------------------------------------------------------------------------------------------------------- |
| `slope_weight`      | Roads become more curvy; they will go around hills rather than over them.                                       |
| `road_reuse_bonus`  | Increases consolidation of roads. Fewer total distinct paths, but thicker "highways".                           |
| `connection_radius` | POIs connect to neighbors further away. Creates a more connected, mesh-like network.                            |
| `poi_density`       | Generates more settlements. High density results in local roads; low density results in long-distance highways. |
| `mask_size`         | Improves pathfinding speed on flat terrain, but may reduce precision on complex slopes.                         |

## How It Works

1.  **Input Analysis:** The heightmap is loaded, and Sobel filters calculate a slope map for the entire terrain.
2.  **Settlement Generation:** The algorithm scans for local height maxima to determine candidate locations. These are clustered into Points of Interest (POIs) with weights based on elevation and prominence.
3.  **Network Construction (A\*):**
    - For every POI, the algorithm finds neighbors within `connection_radius`.
    - A pathfinding search runs between neighbors using **Anisotropic A\***.
    - The cost function prefers flat ground and rewards overlapping with previously generated paths (`road_reuse_bonus`).
4.  **Usage Tracking:** Every time a path is laid down, the "usage" counter for the pixels along that path increases. High usage translates to wider road types.
5.  **Smoothing:** The pixel-perfect A\* path is simplified (RDP algorithm) to remove jitter, then converted into a smooth Catmull-Rom spline.
6.  **Rendering:** The final roads and POIs are drawn to an image buffer, colored by their usage hierarchy.

## Contributing

Contributions are welcome! Areas of interest for future development:

- Water detection (river generation and bridging logic).
- Tunnel generation for steep mountain traversal.
- 3D Mesh export (OBJ/GLTF).
- GUI for real-time parameter tuning.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Pathfinding masks concept based on work by Galin et al.
- Uses the `image`, `rayon`, and `serde` Rust crates.
