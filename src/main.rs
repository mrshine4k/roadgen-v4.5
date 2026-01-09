use image::{GenericImageView, ImageBuffer, Rgb, RgbImage};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::Path;

// ===== CONFIGURATION =====

#[derive(Deserialize)]
struct Config {
    input: InputConfig,
    processing: ProcessingConfig,
    pathfinding: PathfindingConfig,
    visualization: VisConfig,
    random: RandomConfig,
}

#[derive(Deserialize)]
struct InputConfig {
    heightmap_path: String,
    output_dir: String,
}

#[derive(Deserialize)]
struct ProcessingConfig {
    max_road_width: u8,
    poi_min_size: usize,
    poi_max_size: usize,
    connection_radius: f32,
    slope_weight: f32,
    road_reuse_bonus: f32,
}

#[derive(Deserialize)]
struct PathfindingConfig {
    simplify_epsilon: f32,
    max_connections_per_poi: usize,
    mask_size: i32,
}

#[derive(Deserialize)]
struct VisConfig {
    heightmap_color: [u8; 3],
    road_colors: Vec<[u8; 3]>,
    poi_color: [u8; 3],
    spline_color: [u8; 3],
    draw_splines: bool,
}

#[derive(Deserialize)]
struct RandomConfig {
    seed: u64,
    poi_density: f64,
}

// ===== DATA STRUCTURES =====

#[derive(Clone)]
struct Poi {
    x: u32,
    y: u32,
    size: usize,
    weight: f32,
}

struct SpatialGrid {
    cells: Vec<Vec<usize>>,
    cell_size: u32,
    grid_width: usize,
    grid_height: usize,
}

impl SpatialGrid {
    fn new(pois: &[Poi], width: u32, height: u32, cell_size: u32) -> Self {
        let cell_size_usize = cell_size as usize;
        let width_usize = width as usize;
        let height_usize = height as usize;

        let grid_width = width_usize.div_ceil(cell_size_usize);
        let grid_height = height_usize.div_ceil(cell_size_usize);
        let mut cells = vec![Vec::new(); grid_width * grid_height];

        for (i, poi) in pois.iter().enumerate() {
            let gx = (poi.x as usize) / cell_size_usize;
            let gy = (poi.y as usize) / cell_size_usize;
            if gx < grid_width && gy < grid_height {
                cells[gy * grid_width + gx].push(i);
            }
        }

        Self {
            cells,
            cell_size,
            grid_width,
            grid_height,
        }
    }

    fn get_nearby_pois(&self, x: u32, y: u32, radius: f32) -> Vec<usize> {
        let mut result = Vec::new();
        let gx = (x / self.cell_size) as i32;
        let gy = (y / self.cell_size) as i32;
        let cell_radius = (radius / self.cell_size as f32).ceil() as i32 + 1;
        for dy in -cell_radius..=cell_radius {
            for dx in -cell_radius..=cell_radius {
                let nx = gx + dx;
                let ny = gy + dy;
                if nx >= 0
                    && ny >= 0
                    && (nx as usize) < self.grid_width
                    && (ny as usize) < self.grid_height
                {
                    let idx = (ny as usize) * self.grid_width + (nx as usize);
                    result.extend(&self.cells[idx]);
                }
            }
        }
        result
    }
}

struct RoadPath {
    points: Vec<(u32, u32)>,
    avg_usage: u8,
    width_meters: f32, // e.g., 3.0 for local, 12.0 for highway
}

// ===== MASKS (Galin et al.) =====

fn gcd(a: i32, b: i32) -> i32 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn generate_mask(k: i32) -> Vec<(i32, i32)> {
    let mut mask = Vec::new();
    for i in -k..=k {
        for j in -k..=k {
            if (i != 0 || j != 0) && gcd(i, j) == 1 {
                mask.push((i, j));
            }
        }
    }
    mask
}

// ===== MAIN =====

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config_str = fs::read_to_string("config.toml")?;
    let config: Config = toml::from_str(&config_str)?;
    if config.visualization.road_colors.is_empty() {
        return Err("road_colors cannot be empty".into());
    }

    let img = image::open(&config.input.heightmap_path)?;
    let (width, height) = img.dimensions();
    if width == 0 || height == 0 {
        return Err("Invalid heightmap dimensions".into());
    }

    fs::create_dir_all(&config.input.output_dir)?;

    let heightmap: Vec<f32> = img
        .to_rgb8()
        .pixels()
        .map(|p| p[0] as f32 / 255.0)
        .collect();
    let slopes = compute_slopes(&heightmap, width, height)?;

    let mut rng = Pcg64::seed_from_u64(config.random.seed);
    let pois = generate_pois(&heightmap, width, height, &config, &mut rng)?;
    let mut road_usage = vec![0u8; (width as usize) * (height as usize)];

    let pb = ProgressBar::new(pois.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} POIs")
            .unwrap(),
    );
    let road_paths = build_road_network_with_progress(
        &slopes,
        width,
        height,
        &pois,
        &mut road_usage,
        &config,
        &pb,
    )?;
    pb.finish_with_message(format!("Built {} road paths", road_paths.len()));

    let pb = ProgressBar::new(road_paths.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/red}] {pos}/{len} Paths",
            )
            .unwrap(),
    );
    let smoothed_roads: Vec<_> = road_paths
        .par_iter()
        .map(|rp| {
            let spline = catmull_rom_spline(&rp.points, &config.pathfinding, rp.avg_usage);
            pb.inc(1);
            (spline, rp.avg_usage)
        })
        .collect();
    pb.finish();

    let pb = ProgressBar::new_spinner();
    pb.set_message("Rendering...");
    visualize_results(&config, width, height, &pois, &smoothed_roads)?;
    pb.finish_with_message("Done");

    Ok(())
}

// ===== POI & SLOPE =====

fn generate_pois(
    heightmap: &[f32],
    width: u32,
    height: u32,
    config: &Config,
    rng: &mut Pcg64,
) -> Result<Vec<Poi>, Box<dyn std::error::Error>> {
    let w = width as usize;
    let total = w * (height as usize);
    let expected = (total as f64 * config.random.poi_density) as usize;
    let slopes = compute_slopes(heightmap, width, height)?;
    let mut candidate_pois = Vec::with_capacity(expected);
    for _ in 0..expected {
        let idx = rng.gen_range(0..total);
        let x = (idx % w) as u32;
        let y = (idx / w) as u32;
        if x < 10 || y < 10 || x >= width - 10 || y >= height - 10 {
            continue;
        }
        if is_local_maximum(heightmap, width, height, x, y, 3) {
            let slope = slopes[idx];
            let weight = heightmap[idx] * (1.0 + slope * 0.3);
            let size =
                rng.gen_range(config.processing.poi_min_size..=config.processing.poi_max_size);
            candidate_pois.push(Poi { x, y, size, weight });
        }
    }
    Ok(cluster_pois(candidate_pois, width, height, config))
}

fn is_local_maximum(
    heightmap: &[f32],
    width: u32,
    height: u32,
    x: u32,
    y: u32,
    radius: i32,
) -> bool {
    let w = width as usize;
    let center = heightmap[(y as usize) * w + (x as usize)];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            if nx >= 0 && ny >= 0 && nx < width as i32 && ny < height as i32 {
                let nidx = (ny as usize) * w + (nx as usize);
                if heightmap[nidx] > center {
                    return false;
                }
            }
        }
    }
    true
}

fn compute_slopes(
    heightmap: &[f32],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let w = width as usize;
    let h = height as usize;
    let mut slopes = vec![0.0; w * h];
    slopes.par_iter_mut().enumerate().for_each(|(i, s)| {
        let x = i % w;
        let y = i / w;
        if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
            *s = 0.0;
            return;
        }
        let idx = |dx, dy| (y as isize + dy) as usize * w + (x as isize + dx) as usize;
        let gx = -heightmap[idx(-1, -1)] - 2.0 * heightmap[idx(-1, 0)] - heightmap[idx(-1, 1)]
            + heightmap[idx(1, -1)]
            + 2.0 * heightmap[idx(1, 0)]
            + heightmap[idx(1, 1)];
        let gy = -heightmap[idx(-1, -1)] - 2.0 * heightmap[idx(0, -1)] - heightmap[idx(1, -1)]
            + heightmap[idx(-1, 1)]
            + 2.0 * heightmap[idx(0, 1)]
            + heightmap[idx(1, 1)];
        *s = (gx * gx + gy * gy).sqrt();
    });
    Ok(slopes)
}

fn cluster_pois(pois: Vec<Poi>, width: u32, height: u32, config: &Config) -> Vec<Poi> {
    const GRID_SIZE: u32 = 50;
    let grid_w = (width as usize).div_ceil(GRID_SIZE as usize);
    let grid_h = (height as usize).div_ceil(GRID_SIZE as usize);
    let mut grid = vec![Vec::new(); grid_w * grid_h];

    for poi in pois {
        let gx = (poi.x as usize) / GRID_SIZE as usize;
        let gy = (poi.y as usize) / GRID_SIZE as usize;
        if gx < grid_w && gy < grid_h {
            grid[gy * grid_w + gx].push(poi);
        }
    }
    grid.into_par_iter()
        .filter(|c| !c.is_empty())
        .map(|c| {
            let total_weight: f32 = c.iter().map(|p| p.weight).sum();
            let avg_x = c.iter().map(|p| p.x as f64).sum::<f64>() / c.len() as f64;
            let avg_y = c.iter().map(|p| p.y as f64).sum::<f64>() / c.len() as f64;
            let size = c
                .iter()
                .map(|p| p.size)
                .sum::<usize>()
                .min(config.processing.poi_max_size);
            Poi {
                x: avg_x as u32,
                y: avg_y as u32,
                size,
                weight: total_weight,
            }
        })
        .collect()
}

// ===== ROAD NETWORK BUILDING =====

fn build_road_network_with_progress(
    slopes: &[f32],
    width: u32,
    height: u32,
    pois: &[Poi],
    road_usage: &mut [u8],
    config: &Config,
    pb: &ProgressBar,
) -> Result<Vec<RoadPath>, Box<dyn std::error::Error>> {
    let spatial_grid = SpatialGrid::new(pois, width, height, 512);
    let connections: Vec<Vec<(usize, f32)>> = pois
        .par_iter()
        .enumerate()
        .map(|(i, poi)| {
            let mut neighbors: Vec<(usize, f32)> = spatial_grid
                .get_nearby_pois(poi.x, poi.y, config.processing.connection_radius)
                .into_iter()
                .filter_map(|j| {
                    if i == j {
                        return None;
                    }
                    let other = &pois[j];
                    let dx = poi.x as f32 - other.x as f32;
                    let dy = poi.y as f32 - other.y as f32;
                    let dist_sq = dx * dx + dy * dy;
                    let radius_sq = config.processing.connection_radius.powi(2);
                    if dist_sq <= radius_sq
                        && (poi.size >= other.size || other.size >= config.processing.poi_min_size)
                    {
                        let distance = dist_sq.sqrt();
                        let n_ij = (poi.size as f32 * other.size as f32) / (distance + 1.0).powi(2);
                        Some((j, n_ij))
                    } else {
                        None
                    }
                })
                .collect();
            neighbors.sort_unstable_by(|(_, a), (_, b)| b.total_cmp(a));
            neighbors.truncate(config.pathfinding.max_connections_per_poi);
            neighbors
        })
        .collect();

    let mut road_paths = Vec::new();
    for (i, neighbors) in connections.iter().enumerate() {
        for &(j, _) in neighbors {
            if i < j {
                let path = find_path(
                    &pois[i], &pois[j], slopes, width, height, road_usage, config,
                );
                let usage_before: u32 = path
                    .iter()
                    .filter_map(|&(x, y)| {
                        if x < width && y < height {
                            Some(road_usage[(y as usize) * (width as usize) + (x as usize)] as u32)
                        } else {
                            None
                        }
                    })
                    .sum();
                let avg_usage = if !path.is_empty() {
                    ((usage_before as f32 / path.len() as f32) + 1.0)
                        .min(config.processing.max_road_width as f32) as u8
                } else {
                    1
                };
                for &(x, y) in &path {
                    if x < width && y < height {
                        let idx = (y as usize) * (width as usize) + (x as usize);
                        road_usage[idx] = road_usage[idx]
                            .saturating_add(1)
                            .min(config.processing.max_road_width);
                    }
                }

                // Estimate road type: higher usage = highway
                let is_highway = avg_usage >= 3; // configurable threshold

                // Base widths (in "map units" — you can scale to meters if you have scale info)
                let base_width = if is_highway {
                    8.0 // highway
                } else {
                    3.0 // local road
                };

                // Reduce width on steep slopes (optional realism)
                let max_slope_along_path: f32 = path
                    .iter()
                    .filter_map(|&(x, y)| {
                        if x < width && y < height {
                            let idx = (y as usize) * (width as usize) + (x as usize);
                            Some(slopes[idx])
                        } else {
                            None
                        }
                    })
                    .fold(0.0, f32::max);

                // Steep slopes reduce feasible width
                let slope_penalty = (max_slope_along_path * 5.0).min(1.0); // 0.0–1.0
                let adjusted_width = base_width * (1.0 - 0.5 * slope_penalty); // up to 50% narrower

                let road_width = adjusted_width.max(1.0);

                road_paths.push(RoadPath {
                    points: path,
                    avg_usage,
                    width_meters: road_width,
                });
            }
        }
        pb.inc(1);
    }
    Ok(road_paths)
}

// ===== PATHFINDING (Anisotropic A*) =====

fn find_path(
    start: &Poi,
    end: &Poi,
    slopes: &[f32],
    width: u32,
    height: u32,
    road_usage: &[u8],
    config: &Config,
) -> Vec<(u32, u32)> {
    let alpha = config.processing.road_reuse_bonus;
    let mask = generate_mask(config.pathfinding.mask_size);

    #[derive(Clone)]
    struct Node {
        x: u32,
        y: u32,
        g: f32,
        h: f32,
    }
    #[derive(Clone)]
    struct SearchNode(Node);
    impl PartialEq for SearchNode {
        fn eq(&self, o: &Self) -> bool {
            self.0.x == o.0.x && self.0.y == o.0.y
        }
    }
    impl Eq for SearchNode {}
    impl PartialOrd for SearchNode {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
            Some(self.cmp(o))
        }
    }
    impl Ord for SearchNode {
        fn cmp(&self, o: &Self) -> Ordering {
            let f1 = self.0.g + self.0.h;
            let f2 = o.0.g + o.0.h;
            f2.total_cmp(&f1)
                .then_with(|| self.0.x.cmp(&o.0.x))
                .then_with(|| self.0.y.cmp(&o.0.y))
        }
    }

    let heuristic = |x: u32, y: u32| -> f32 {
        ((x as f32 - end.x as f32).powi(2) + (y as f32 - end.y as f32).powi(2)).sqrt()
    };

    let mut open = BinaryHeap::new();
    let mut closed = HashSet::new();
    let mut came_from = HashMap::new();
    open.push(SearchNode(Node {
        x: start.x,
        y: start.y,
        g: 0.0,
        h: heuristic(start.x, start.y),
    }));

    while let Some(SearchNode(current)) = open.pop() {
        let pos = (current.x, current.y);
        if current.x == end.x && current.y == end.y {
            break;
        }
        if !closed.insert(pos) {
            continue;
        }

        for &(dx, dy) in &mask {
            let nx = current.x as i32 + dx;
            let ny = current.y as i32 + dy;
            if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                continue;
            }
            let nx = nx as u32;
            let ny = ny as u32;
            let next = (nx, ny);
            if closed.contains(&next) {
                continue;
            }

            let segment_cost = calculate_segment_cost(
                current.x, current.y, nx, ny, slopes, width, height, road_usage, config, alpha,
            );
            let g = current.g + segment_cost;
            let h = heuristic(nx, ny);

            if g < *came_from.get(&next).map_or(&f32::INFINITY, |(_, gs)| gs) {
                came_from.insert(next, (pos, g));
                open.push(SearchNode(Node { x: nx, y: ny, g, h }));
            }
        }
    }

    let mut path = Vec::new();
    let mut current = (end.x, end.y);
    while let Some(&(prev, _)) = came_from.get(&current) {
        path.push(current);
        current = prev;
    }
    path.push((start.x, start.y));
    path.reverse();
    if path.len() <= 1 {
        bresenham_line(start.x, start.y, end.x, end.y)
    } else {
        path
    }
}

fn calculate_segment_cost(
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
    slopes: &[f32],
    width: u32,
    height: u32,
    road_usage: &[u8],
    config: &Config,
    alpha: f32,
) -> f32 {
    let w = width as usize;
    let dx = x2 as i32 - x1 as i32;
    let dy = y2 as i32 - y1 as i32;
    let steps = dx.abs().max(dy.abs()).max(1) as usize;
    let mut total_cost = 0.0;
    let mut any_existing = false;

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = (x1 as f32 + dx as f32 * t).round() as u32;
        let y = (y1 as f32 + dy as f32 * t).round() as u32;
        if x < width && y < height {
            let idx = (y as usize) * w + (x as usize);
            let base_cost = 1.0 + slopes[idx] * config.processing.slope_weight;
            let mut cell_cost = 1.0 + (base_cost - 1.0).powi(2);

            // Apply reuse bonus ONLY to this cell if it's on an existing road
            if road_usage[idx] > 0 {
                cell_cost /= config.processing.road_reuse_bonus.max(1.0);
            }

            total_cost += cell_cost;
            if road_usage[idx] > 0 {
                any_existing = true;
            }
        }
    }

    let avg = total_cost / (steps + 1) as f32;
    if any_existing {
        avg / alpha.max(0.5)
    } else {
        avg
    }
}

fn bresenham_line(x0: u32, y0: u32, x1: u32, y1: u32) -> Vec<(u32, u32)> {
    let (mut x, mut y) = (x0 as i32, y0 as i32);
    let (x1, y1) = (x1 as i32, y1 as i32);
    let dx = (x1 - x).abs();
    let dy = (y1 - y).abs();
    let sx = if x < x1 { 1 } else { -1 };
    let sy = if y < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut points = vec![(x as u32, y as u32)];
    loop {
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
        points.push((x as u32, y as u32));
    }
    points
}

// ===== SPLINES =====

fn catmull_rom_spline(path: &[(u32, u32)], cfg: &PathfindingConfig, usage: u8) -> Vec<(f32, f32)> {
    if path.len() < 2 {
        return path.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
    }
    let adaptive_epsilon = cfg.simplify_epsilon + (usage as f32) * 0.5;
    let simplified = rdp_simplify(path, adaptive_epsilon);
    if simplified.len() < 2 {
        return simplified
            .iter()
            .map(|&(x, y)| (x as f32, y as f32))
            .collect();
    }

    let mut points: Vec<(f32, f32)> = simplified
        .iter()
        .map(|&(x, y)| (x as f32, y as f32))
        .collect();
    if points.len() > 1 {
        let dx0 = points[0].0 - points[1].0;
        let dy0 = points[0].1 - points[1].1;
        points.insert(0, (points[0].0 + dx0, points[0].1 + dy0));
        let n = points.len() - 1;
        let dxn = points[n].0 - points[n - 1].0;
        let dyn_val = points[n].1 - points[n - 1].1;
        points.push((points[n].0 + dxn, points[n].1 + dyn_val));
    }

    let mut spline = Vec::new();
    for i in 1..points.len() - 2 {
        for t in 0..10 {
            let t = t as f32 * 0.1;
            let p0 = points[i - 1];
            let p1 = points[i];
            let p2 = points[i + 1];
            let p3 = points[i + 2];
            let t2 = t * t;
            let t3 = t2 * t;
            let x = 0.5
                * (2.0 * p1.0
                    + (-p0.0 + p2.0) * t
                    + (2.0 * p0.0 - 5.0 * p1.0 + 4.0 * p2.0 - p3.0) * t2
                    + (-p0.0 + 3.0 * p1.0 - 3.0 * p2.0 + p3.0) * t3);
            let y = 0.5
                * (2.0 * p1.1
                    + (-p0.1 + p2.1) * t
                    + (2.0 * p0.1 - 5.0 * p1.1 + 4.0 * p2.1 - p3.1) * t2
                    + (-p0.1 + 3.0 * p1.1 - 3.0 * p2.1 + p3.1) * t3);
            spline.push((x, y));
        }
    }
    spline
}

fn rdp_simplify(path: &[(u32, u32)], epsilon: f32) -> Vec<(u32, u32)> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let first = path[0];
    let last = *path.last().unwrap();
    let line_dx = last.0 as f32 - first.0 as f32;
    let line_dy = last.1 as f32 - first.1 as f32;
    let line_len_sq = line_dx * line_dx + line_dy * line_dy;

    let mut max_dist_sq = 0.0;
    let mut max_idx = 0;
    for (i, &point) in path.iter().enumerate().skip(1).take(path.len() - 2) {
        let px = point.0 as f32 - first.0 as f32;
        let py = point.1 as f32 - first.1 as f32;
        let dist_sq = if line_len_sq < f32::EPSILON {
            px * px + py * py
        } else {
            let u = (px * line_dx + py * line_dy) / line_len_sq;
            let ix = first.0 as f32 + u * line_dx;
            let iy = first.1 as f32 + u * line_dy;
            (point.0 as f32 - ix).powi(2) + (point.1 as f32 - iy).powi(2)
        };
        if dist_sq > max_dist_sq {
            max_dist_sq = dist_sq;
            max_idx = i;
        }
    }

    if max_dist_sq > epsilon * epsilon {
        let left = rdp_simplify(&path[..=max_idx], epsilon);
        let right = rdp_simplify(&path[max_idx..], epsilon);
        let mut result = Vec::with_capacity(left.len() + right.len());
        let left_len = left.len();
        result.extend(left.into_iter().take(left_len.saturating_sub(1)));
        result.extend(right);
        result
    } else {
        vec![first, last]
    }
}

// ===== VISUALIZATION =====

fn get_next_output_path(
    output_dir: &str,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let dir = Path::new(output_dir);
    for counter in 1..=9999 {
        let path = dir.join(format!("road_network_{:04}.png", counter));
        if !path.exists() {
            return Ok(path);
        }
    }
    Err("Too many output files".into())
}

fn visualize_results(
    config: &Config,
    width: u32,
    height: u32,
    pois: &[Poi],
    smoothed_roads: &[(Vec<(f32, f32)>, u8)],
) -> Result<(), Box<dyn std::error::Error>> {
    let base_color = Rgb(config.visualization.heightmap_color);
    let mut img = ImageBuffer::from_fn(width, height, |_, _| base_color);

    // Unified disc drawing
    let draw_disc = |img: &mut RgbImage, cx: i32, cy: i32, radius: f32, color: Rgb<u8>| {
        let r2 = (radius * radius) as i32;
        let w = img.width() as i32;
        let h = img.height() as i32;
        let r = radius.ceil() as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r2 {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x >= 0 && x < w && y >= 0 && y < h {
                        img.put_pixel(x as u32, y as u32, color);
                    }
                }
            }
        }
    };

    let mut smooth_roads_sorted = smoothed_roads.to_vec();
    smooth_roads_sorted.sort_by_key(|k| k.1);

    for (spline, usage) in &smooth_roads_sorted {
        if spline.is_empty() {
            continue;
        }
        let color_idx =
            (usage.saturating_sub(1) as usize).min(config.visualization.road_colors.len() - 1);
        let color = Rgb(config.visualization.road_colors[color_idx]);

        for i in 0..spline.len() {
            let (x, y) = spline[i];
            draw_disc(
                &mut img,
                x.round() as i32,
                y.round() as i32,
                *usage as f32 * 2.0,
                color,
            );
            if i + 1 < spline.len() {
                let (nx, ny) = spline[i + 1];
                draw_thick_line(
                    &mut img,
                    x.round() as i32,
                    y.round() as i32,
                    nx.round() as i32,
                    ny.round() as i32,
                    *usage as f32 * 2.0,
                    color,
                );
            }
        }
    }

    for poi in pois {
        draw_disc(
            &mut img,
            poi.x as i32,
            poi.y as i32,
            (poi.size / 2) as f32,
            Rgb(config.visualization.poi_color),
        );
    }

    if config.visualization.draw_splines {
        let spline_color = Rgb(config.visualization.spline_color);
        for (spline, _) in smoothed_roads {
            for i in 0..spline.len().saturating_sub(1) {
                let (x0, y0) = spline[i];
                let (x1, y1) = spline[i + 1];
                draw_line(
                    &mut img,
                    x0.round() as i32,
                    y0.round() as i32,
                    x1.round() as i32,
                    y1.round() as i32,
                    spline_color,
                );
            }
        }
    }

    let output_path = get_next_output_path(&config.input.output_dir)?;
    img.save(&output_path)?;
    Ok(())
}

// Thick and thin line drawing (kept separate — different logic)

fn draw_thick_line(
    img: &mut RgbImage,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    radius: f32,
    color: Rgb<u8>,
) {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let (mut x, mut y) = (x0, y0);
    loop {
        // Reuse unified draw_disc
        let r2 = (radius * radius) as i32;
        let w = img.width() as i32;
        let h = img.height() as i32;
        let r = radius.ceil() as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r2 {
                    let xx = x + dx;
                    let yy = y + dy;
                    if xx >= 0 && xx < w && yy >= 0 && yy < h {
                        img.put_pixel(xx as u32, yy as u32, color);
                    }
                }
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_line(img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>) {
    let dx = (x1 - x0).abs();
    let dy = (y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx - dy;
    let (mut x, mut y) = (x0, y0);
    let w = img.width() as i32;
    let h = img.height() as i32;
    loop {
        if x >= 0 && x < w && y >= 0 && y < h {
            img.put_pixel(x as u32, y as u32, color);
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}
