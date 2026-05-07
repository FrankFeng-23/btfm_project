// src/main.rs
use anyhow::{anyhow, Result};
use chrono::{NaiveDate, Datelike};
use env_logger::{Builder, Env};
use glob::glob;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{info, warn, error};
use memmap2::MmapOptions;
use ndarray::{Array2, Array3};
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::panic::AssertUnwindSafe;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};
use structopt::StructOpt;
use tiff::decoder::{Decoder, DecodingResult, Limits};

#[derive(Debug, StructOpt)]
#[structopt(name = "process_tile_downstream_wo_json", about = "Process Sentinel-2 data for a single MGRS tile")]
struct Opt {
    /// Input directory (where raw tiff files are organized in band folders)
    #[structopt(short = "i", long = "input", parse(from_os_str))]
    input_dir: PathBuf,

    /// Output directory (where processed NPY files will go)
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output_dir: PathBuf,

    /// Downsample rate (default=10)
    #[structopt(short = "r", long = "sample-rate", default_value = "10")]
    sample_rate: usize,

    /// Number of threads (default=10) to use for parallel tasks
    #[structopt(short = "n", long = "num-threads", default_value = "10")]
    num_threads: usize,
    
    /// Number of time slices to process in parallel
    #[structopt(short = "b", long = "batch-size", default_value = "5")]
    batch_size: usize,
    
    /// Cache strategy (0=minimal, 1=moderate, 2=aggressive)
    #[structopt(short = "c", long = "cache-level", default_value = "1")]
    cache_level: u8,
}

/// Define the 10 bands we're interested in.
/// Output channel order is fixed by Tessera's pretraining/checkpoint convention:
/// RGB-NIR first (B04, B02, B03, B08), then narrow NIR (B8A),
/// then three red-edge (B05, B06, B07), then two SWIR (B11, B12).
/// S2_BAND_MEAN/S2_BAND_STD in tessera_infer/.../ssl_dataset.py are aligned to this order.
const BANDS: [&str; 10] = [
    "red",
    "blue",
    "green",
    "nir",
    "nir08",
    "rededge1",
    "rededge2",
    "rededge3",
    "swir16",
    "swir22",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandResolution {
    Full, // 10m
    Half, // 20m
}

impl BandResolution {
    fn label(&self) -> &'static str {
        match self {
            BandResolution::Full => "10m",
            BandResolution::Half => "20m",
        }
    }
}

/// Custom struct to replace the JSON metadata
#[derive(Debug, Clone)]
struct TileData {
    date: NaiveDate,
    doy: u16,
    filename: String,
}

/// Result of processing a single time slice
#[derive(Debug)]
struct ProcessedSlice {
    index: usize,
    bands_data: Array3<u16>,
    mask_data: Array2<u8>,
    band_sums: Vec<f64>,
    band_sqsums: Vec<f64>,
    doy: u16,
}

/// Cache structure for band data
struct BandCache {
    enabled: bool,
    data: HashMap<String, Arc<Array2<u16>>>,
}

impl BandCache {
    fn new(enabled: bool) -> Self {
        BandCache {
            enabled,
            data: HashMap::new(),
        }
    }
    
    fn get(&self, key: &str) -> Option<Arc<Array2<u16>>> {
        if !self.enabled {
            return None;
        }
        self.data.get(key).cloned()
    }
    
    fn insert(&mut self, key: String, value: Array2<u16>) -> Arc<Array2<u16>> {
        if !self.enabled {
            return Arc::new(value);
        }
        let arc_value = Arc::new(value);
        self.data.insert(key, arc_value.clone());
        arc_value
    }
}

fn main() -> Result<()> {
    let start_time = Instant::now();
    
    // Initialize env_logger
    let mut builder = Builder::from_env(Env::default().default_filter_or("info"));
    builder.format_timestamp_secs();
    builder.init();

    let opt = Opt::from_args();

    // Configure Rayon to use specified number of threads
    ThreadPoolBuilder::new()
        .num_threads(opt.num_threads)
        .build_global()
        .map_err(|e| anyhow!("Failed to build thread pool: {:?}", e))?;

    info!("Starting process_tile_downstream_wo_json with optimized parallel processing...");
    info!("Input dir: {:?}", opt.input_dir);
    info!("Output dir: {:?}", opt.output_dir);
    info!("Sample rate: {}", opt.sample_rate);
    info!("Num threads: {}", opt.num_threads);
    info!("Batch size: {}", opt.batch_size);
    info!("Cache level: {}", opt.cache_level);

    // Determine caching strategy based on cache level
    let use_band_cache = opt.cache_level >= 1;
    let full_prefetch_bands = opt.cache_level >= 2;
    
    info!("Cache settings - Band cache: {}, Full prefetch: {}", use_band_cache, full_prefetch_bands);

    // Get full resolution dimensions from a sample TIFF
    let (full_height, full_width) = get_full_resolution(&opt.input_dir)?;
    info!("Detected full resolution: height = {}, width = {}", full_height, full_width);

    // Calculate output dimensions
    let output_height = full_height / opt.sample_rate;
    let output_width = full_width / opt.sample_rate;
    let num_bands = BANDS.len();

    // Create output directory if it doesn't exist
    fs::create_dir_all(&opt.output_dir)?;

    // Initialize progress bars
    let multi_progress = MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("#>-");
    
    let loading_pb = multi_progress.add(ProgressBar::new(0));
    loading_pb.set_style(style.clone());
    loading_pb.set_message("Scanning and validating files...");
    
    // Build metadata from TIFF filenames and validate all required bands exist
    let tile_data = scan_and_validate_tiff_files(&opt.input_dir, &loading_pb)?;
    loading_pb.finish_with_message("File scanning and validation done.");

    if tile_data.is_empty() {
        warn!("No valid TIFF file sets found in subdirectories of {:?}. Exiting early.", opt.input_dir);
        return Ok(());
    }
    
    let band_resolutions = detect_band_resolutions(&opt.input_dir, full_height, full_width)?;
    for (band_name, resolution) in BANDS.iter().zip(band_resolutions.iter()) {
        info!("Detected band resolution for {}: {}", band_name, resolution.label());
    }

    let num_time_slices = tile_data.len();
    info!("Found {} valid time slices to process", num_time_slices);

    // Create a shared cache for band data
    let band_cache = Arc::new(Mutex::new(BandCache::new(use_band_cache)));
    
    // Prefetch band files if aggressive caching is enabled
    if full_prefetch_bands {
        info!("Prefetching band files into cache...");
        let prefetch_pb = multi_progress.add(ProgressBar::new((num_time_slices * num_bands) as u64));
        prefetch_pb.set_style(style.clone());
        prefetch_pb.set_message("Prefetching band files...");
        
        for (_band_idx, band_name) in BANDS.iter().enumerate() {
            for tile in &tile_data {
                let filename_base = &tile.filename;
                let band_path = opt.input_dir.join(band_name).join(format!("{}", filename_base));
                
                if band_path.exists() {
                    let cache_key = band_path.to_string_lossy().to_string();
                    match read_tiff_safe(&band_path) {
                        Ok(data) => {
                            let mut cache = band_cache.lock();
                            cache.insert(cache_key, data);
                        },
                        Err(e) => {
                            warn!("Error prefetching band file {:?}: {}", band_path, e);
                        }
                    }
                }
                prefetch_pb.inc(1);
            }
        }
        prefetch_pb.finish_with_message("Band file prefetching complete");
    }

    // Progress bar for time slice processing
    let total_steps = num_time_slices as u64;
    let pb_slices = multi_progress.add(ProgressBar::new(total_steps));
    pb_slices.set_style(style.clone());
    pb_slices.set_message("Processing time slices...");

    // Process time slices in batches for better parallelism
    let batches = (0..num_time_slices).collect::<Vec<_>>()
        .chunks(opt.batch_size)
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    
    info!("Processing {} time slices in {} batches of size {}", 
          num_time_slices, batches.len(), opt.batch_size);
    
    let mut successful_slices = Vec::new();
    
    for batch in batches {
        let results: Vec<Option<ProcessedSlice>> = batch.par_iter().map(|&slice_idx| {
            // Wrap entire time slice processing in panic catching
            let slice_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<ProcessedSlice> {
                process_time_slice_safe(
                    slice_idx,
                    &tile_data[slice_idx],
                    &opt.input_dir,
                    &band_cache,
                    use_band_cache,
                    full_height,
                    full_width,
                    output_height,
                    output_width,
                    opt.sample_rate,
                    num_bands,
                    &band_resolutions,
                )
            }));
            
            match slice_result {
                Ok(Ok(slice)) => Some(slice),
                Ok(Err(e)) => {
                    error!("Error processing time slice {}: {}", slice_idx, e);
                    None
                },
                Err(_) => {
                    error!("Panic occurred while processing time slice {}. Skipping this timestep.", slice_idx);
                    None
                }
            }
        }).collect();
        
        // Collect successful results
        for result in results {
            if let Some(slice) = result {
                successful_slices.push(slice);
            }
            pb_slices.inc(1);
        }
    }

    pb_slices.finish_with_message("All time slices processed.");

    let num_successful = successful_slices.len();
    info!("Successfully processed {} out of {} time slices", num_successful, num_time_slices);

    if num_successful == 0 {
        error!("No time slices were successfully processed!");
        return Err(anyhow!("No valid data to write"));
    }

    // Sort successful slices by their original index to maintain temporal order
    successful_slices.sort_by_key(|slice| slice.index);

    // Now create NPY files with correct headers based on actual successful count
    let band_file_path = opt.output_dir.join("bands.npy");
    let mut band_file = create_npy_file::<u16, 4>(
        &band_file_path,
        &[
            num_successful,  // Use actual successful count
            output_height,
            output_width,
            num_bands
        ],
    )?;
    info!("Created output bands file: {:?} with shape [{}, {}, {}, {}]", 
          band_file_path, num_successful, output_height, output_width, num_bands);

    let mask_file_path = opt.output_dir.join("masks.npy");
    let mut mask_file = create_npy_file::<u8, 3>(
        &mask_file_path,
        &[
            num_successful,  // Use actual successful count
            output_height,
            output_width
        ],
    )?;
    info!("Created output masks file: {:?} with shape [{}, {}, {}]", 
          mask_file_path, num_successful, output_height, output_width);

    // Initialize band statistics accumulators
    let mut sum_bands = vec![0.0; num_bands];
    let mut sum_sq_bands = vec![0.0; num_bands];

    // Write all successful slices to files and accumulate statistics
    let mut successful_doys = Vec::with_capacity(num_successful);
    
    for slice in &successful_slices {
        // Write slice data to band_file
        let slice_bytes = unsafe {
            std::slice::from_raw_parts(
                slice.bands_data.as_ptr() as *const u8,
                slice.bands_data.len() * std::mem::size_of::<u16>(),
            )
        };
        if let Err(e) = band_file.write_all(slice_bytes) {
            error!("Error writing band data: {}", e);
            return Err(anyhow!("Failed to write band data: {}", e));
        }
        
        // Write mask slice
        let mask_bytes = unsafe {
            std::slice::from_raw_parts(
                slice.mask_data.as_ptr() as *const u8,
                slice.mask_data.len() * std::mem::size_of::<u8>(),
            )
        };
        if let Err(e) = mask_file.write_all(mask_bytes) {
            error!("Error writing mask data: {}", e);
            return Err(anyhow!("Failed to write mask data: {}", e));
        }
        
        // Update global statistics
        for b in 0..num_bands {
            sum_bands[b] += slice.band_sums[b];
            sum_sq_bands[b] += slice.band_sqsums[b];
        }
        
        // Track successful DOYs
        successful_doys.push(slice.doy);
    }

    // Flush files to ensure data is written to disk
    band_file.flush()?;
    mask_file.flush()?;

    // Write day_of_years to doys.npy (only successful ones)
    write_npy_simple::<u16, 1>(
        &opt.output_dir.join("doys.npy"),
        &[num_successful],
        &successful_doys,
    )?;

    // Calculate and write band statistics
    let total_pixel_count = (num_successful * output_height * output_width) as f64;
    let mut band_means = Vec::with_capacity(num_bands);
    let mut band_stds = Vec::with_capacity(num_bands);
    for b in 0..num_bands {
        let mean = sum_bands[b] / total_pixel_count;
        let var = sum_sq_bands[b] / total_pixel_count - mean * mean;
        let std = var.sqrt().max(0.0);
        band_means.push(mean);
        band_stds.push(std);
    }

    // Write band_mean.npy and band_std.npy
    write_npy_simple::<f64, 1>(
        &opt.output_dir.join("band_mean.npy"),
        &[num_bands],
        &band_means,
    )?;
    write_npy_simple::<f64, 1>(
        &opt.output_dir.join("band_std.npy"),
        &[num_bands],
        &band_stds,
    )?;

    // Write info about successful slices
    let info_content = format!(
        "Original time slices: {}\nSuccessful time slices: {}\nSkipped time slices: {}\n",
        num_time_slices,
        num_successful,
        num_time_slices - num_successful
    );
    std::fs::write(opt.output_dir.join("processing_info.txt"), info_content)?;

    let elapsed = start_time.elapsed();
    info!("All done! Processing completed in {:.2} seconds", elapsed.as_secs_f64());
    info!("Output in {:?}", opt.output_dir);
    
    Ok(())
}

/// Safe wrapper for processing a single time slice with comprehensive error handling
fn process_time_slice_safe(
    slice_idx: usize,
    tile: &TileData,
    input_dir: &Path,
    band_cache: &Arc<Mutex<BandCache>>,
    use_band_cache: bool,
    full_height: usize,
    full_width: usize,
    output_height: usize,
    output_width: usize,
    sample_rate: usize,
    num_bands: usize,
    band_resolutions: &[BandResolution],
) -> Result<ProcessedSlice> {
    // Double-wrapped panic catching for maximum safety
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<ProcessedSlice> {
        process_time_slice(
            slice_idx,
            tile,
            input_dir,
            band_cache,
            use_band_cache,
            full_height,
            full_width,
            output_height,
            output_width,
            sample_rate,
            num_bands,
            band_resolutions,
        )
    }));
    
    match result {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Panic occurred during time slice processing for index {}", slice_idx)),
    }
}

/// Process a single time slice with comprehensive error handling
fn process_time_slice(
    slice_idx: usize,
    tile: &TileData,
    input_dir: &Path,
    band_cache: &Arc<Mutex<BandCache>>,
    use_band_cache: bool,
    full_height: usize,
    full_width: usize,
    output_height: usize,
    output_width: usize,
    sample_rate: usize,
    num_bands: usize,
    band_resolutions: &[BandResolution],
) -> Result<ProcessedSlice> {
    let filename_base = &tile.filename;
    
    // Process all bands for this time slice in parallel with enhanced error handling
    let band_results: Result<Vec<(usize, Array2<u16>)>> =
        (0..num_bands)
            .into_par_iter()
            .map(|band_idx| -> Result<(usize, Array2<u16>)> {
                // Wrap band processing in panic catching
                let band_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, Array2<u16>)> {
                    let band_name = BANDS[band_idx];
                    let band_path = input_dir.join(band_name).join(format!("{}", filename_base));
                    
                    if !band_path.exists() {
                        return Err(anyhow!("Band file does not exist: {:?}", band_path));
                    }
                    
                    // Try to get from cache first
                    let cache_key = band_path.to_string_lossy().to_string();
                    let cache = band_cache.lock();
                    
                    if let Some(cached_data) = cache.get(&cache_key) {
                        // Use cached data
                        drop(cache); // Release lock before processing
                        let processed = process_band(
                            &cached_data,
                            full_height,
                            full_width,
                            sample_rate,
                            band_resolutions[band_idx],
                            band_name,
                        );
                        return Ok((band_idx, processed));
                    }
                    drop(cache); // Release lock if not found in cache
                    
                    // Read and process the band
                    let data = read_tiff_safe(&band_path)
                        .map_err(|e| anyhow!("Failed to read band file {:?}: {}", band_path, e))?;
                    
                    // Cache the raw data if enabled
                    if use_band_cache {
                        let mut cache = band_cache.lock();
                        cache.insert(cache_key, data.clone());
                    }
                    
                    let processed = process_band(
                        &data,
                        full_height,
                        full_width,
                        sample_rate,
                        band_resolutions[band_idx],
                        band_name,
                    );
                    Ok((band_idx, processed))
                }));
                
                match band_result {
                    Ok(result) => result,
                    Err(_) => Err(anyhow!("Panic occurred while processing band {} for slice {}", band_idx, slice_idx)),
                }
            })
            .collect();
            
    let band_results = band_results?;
    
    // Optimize: Create bands array with known dimensions
    let mut slice_data = Array3::<u16>::zeros((output_height, output_width, num_bands));
    
    // Merge band results into a 3D array using optimized assignment
    for (band_idx, band_array) in band_results {
        for i in 0..output_height {
            for j in 0..output_width {
                slice_data[[i, j, band_idx]] = band_array[[i, j]];
            }
        }
    }
    
    // Calculate band statistics
    let mut band_sums = vec![0.0; num_bands];
    let mut band_sqsums = vec![0.0; num_bands];
    
    for i in 0..output_height {
        for j in 0..output_width {
            for b in 0..num_bands {
                let val = slice_data[[i, j, b]] as f64;
                band_sums[b] += val;
                band_sqsums[b] += val * val;
            }
        }
    }
    
    // Process mask (SCL data if available) with enhanced error handling
    let mut mask_slice = Array2::<u8>::zeros((output_height, output_width));
    
    // Try to find SCL data with panic protection
    let scl_path = input_dir.join("scl").join(format!("{}", filename_base));
    if scl_path.exists() {
        // Wrap SCL processing in panic catching
        let scl_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array2<u8>> {
            // Try SCL cache first
            let cache_key = scl_path.to_string_lossy().to_string();
            let cache = band_cache.lock();
            
            if let Some(cached_scl) = cache.get(&cache_key) {
                // Use cached SCL data
                drop(cache);
                let mask_low_res = cached_scl.mapv(|px: u16| {
                    if matches!(px, 0 | 1 | 2 | 3 | 8 | 9) {
                        0u8
                    } else {
                        1u8
                    }
                });
                let mask_full = resample_mask(&mask_low_res, full_height, full_width);
                return Ok(downsample_mask(&mask_full, full_height, full_width, sample_rate));
            }
            drop(cache);
            
            match read_tiff_safe(&scl_path) {
                Ok(scl_data) => {
                    // Cache SCL data if enabled
                    if use_band_cache {
                        let mut cache = band_cache.lock();
                        cache.insert(cache_key, scl_data.clone());
                    }
                    
                    let mask_low_res = scl_data.mapv(|px: u16| {
                        if matches!(px, 0 | 1 | 2 | 3 | 8 | 9) {
                            0u8
                        } else {
                            1u8
                        }
                    });
                    let mask_full = resample_mask(&mask_low_res, full_height, full_width);
                    Ok(downsample_mask(&mask_full, full_height, full_width, sample_rate))
                },
                Err(e) => Err(anyhow!("Error reading SCL file {:?}: {}", scl_path, e))
            }
        }));
        
        match scl_result {
            Ok(Ok(mask)) => {
                mask_slice = mask;
            },
            Ok(Err(e)) => {
                warn!("Error processing SCL file {:?}: {}", scl_path, e);
                mask_slice.fill(1u8);
            },
            Err(_) => {
                warn!("Panic occurred while processing SCL file {:?}. Using default mask.", scl_path);
                mask_slice.fill(1u8);
            }
        }
    } else {
        // If SCL is missing, default to all valid
        mask_slice.fill(1u8);
    }
    
    Ok(ProcessedSlice {
        index: slice_idx,
        bands_data: slice_data,
        mask_data: mask_slice,
        band_sums,
        band_sqsums,
        doy: tile.doy,
    })
}

/// Scan TIFF files in band subdirectories and validate that all required bands exist
fn scan_and_validate_tiff_files(input_dir: &Path, pb: &ProgressBar) -> Result<Vec<TileData>> {
    let start = Instant::now();
    
    // We'll use the "red" subdirectory as the reference for available dates
    let red_dir = input_dir.join("red");
    let pattern = red_dir.join("*.tiff").to_string_lossy().to_string();
    info!("Scanning for TIFF files with pattern: {}", pattern);
    
    let mut files = Vec::new();
    for entry in glob(&pattern)? {
        match entry {
            Ok(path) => files.push(path),
            Err(e) => {
                warn!("Bad path from glob: {:?}", e);
            }
        }
    }
    
    if files.is_empty() {
        warn!("No TIFF files found in {:?}", red_dir);
        return Ok(Vec::new());
    }
    
    pb.set_length(files.len() as u64);
    info!("Found {} TIFF files in red directory", files.len());
    
    // Extract date information from filenames and validate band completeness
    let tile_data: Vec<Option<TileData>> = files.par_iter().map(|path| {
        pb.inc(1);
        
        // Wrap validation in panic catching
        let validation_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Option<TileData> {
            if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                // Parse date from filename (format: YYYY-MM-DD_mosaic.tiff)
                if let Some(date_str) = filename.split('_').next() {
                    match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                        Ok(date) => {
                            let doy = date.ordinal() as u16;
                            
                            // Validate that all required bands exist and are readable
                            let mut all_bands_valid = true;
                            for band_name in &BANDS {
                                let band_path = input_dir.join(band_name).join(filename);
                                if !band_path.exists() {
                                    warn!("Missing band {} for date {}: {:?}", band_name, date_str, band_path);
                                    all_bands_valid = false;
                                    break;
                                }
                                
                                // Quick validation - try to read TIFF header
                                if let Err(e) = validate_tiff_file(&band_path) {
                                    warn!("Invalid TIFF file {} for date {}: {}", band_name, date_str, e);
                                    all_bands_valid = false;
                                    break;
                                }
                            }
                            
                            if all_bands_valid {
                                return Some(TileData {
                                    date,
                                    doy,
                                    filename: filename.to_string(),
                                });
                            } else {
                                warn!("Skipping date {} due to incomplete or invalid bands", date_str);
                                return None;
                            }
                        },
                        Err(e) => {
                            warn!("Failed to parse date from filename {}: {}", filename, e);
                            return None;
                        }
                    }
                }
            }
            None
        }));
        
        match validation_result {
            Ok(result) => result,
            Err(_) => {
                warn!("Panic occurred during validation of file {:?}. Skipping.", path);
                None
            }
        }
    }).collect();
    
    // Filter out None values and collect valid tiles
    let mut valid_tiles: Vec<TileData> = tile_data.into_iter()
        .filter_map(|opt| opt)
        .collect();
    
    // Sort by date (which will also sort by DOY)
    valid_tiles.sort_by(|a, b| a.date.cmp(&b.date));
    
    let elapsed = start.elapsed();
    pb.finish_with_message(format!("Found {} valid and complete time slices in {:.2}s", valid_tiles.len(), elapsed.as_secs_f64()));
    Ok(valid_tiles)
}

/// Validate that a TIFF file can be opened and read without errors
fn validate_tiff_file(path: &Path) -> Result<()> {
    // Triple-wrapped panic catching for maximum safety
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        let inner_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<()> {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => return Err(anyhow!("Failed to open {:?}: {}", path, e)),
            };
            
            // Use memory mapping for efficiency
            let mmap = match unsafe { MmapOptions::new().map(&file) } {
                Ok(m) => m,
                Err(e) => return Err(anyhow!("Failed to mmap {:?}: {}", path, e)),
            };
            
            // Wrap decoder operations
            let decode_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<()> {
                let cursor = std::io::Cursor::new(&mmap);
                let decoder_result = Decoder::new(cursor);
                
                let mut decoder = match decoder_result {
                    Ok(d) => d,
                    Err(e) => return Err(anyhow!("Failed to create decoder for {:?}: {}", path, e)),
                };
                
                decoder = decoder.with_limits(Limits::unlimited());
                
                // Try to get dimensions - this will fail if the TIFF is corrupted
                match decoder.dimensions() {
                    Ok(_) => Ok(()),
                    Err(e) => Err(anyhow!("Failed to read dimensions from {:?}: {}", path, e)),
                }
            }));
            
            match decode_result {
                Ok(result) => result,
                Err(_) => Err(anyhow!("Decoder panic while validating {:?}", path)),
            }
        }));
        
        match inner_result {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Inner panic while validating {:?}", path)),
        }
    }));
    
    match result {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Outer panic while validating TIFF {:?}", path)),
    }
}

/// Read a TIFF file using memory mapping with enhanced error handling
fn read_tiff_safe(path: &Path) -> Result<Array2<u16>> {
    // Triple-wrapped panic catching to handle all possible panics
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array2<u16>> {
        let inner_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array2<u16>> {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => return Err(anyhow!("Failed to open {:?}: {}", path, e)),
            };
            
            // Use memory mapping for efficiency
            let mmap = match unsafe { MmapOptions::new().map(&file) } {
                Ok(m) => m,
                Err(e) => return Err(anyhow!("Failed to mmap {:?}: {}", path, e)),
            };
            
            // Wrap decoder operations in another catch_unwind for extra safety
            let decode_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array2<u16>> {
                // Create decoder with explicit error handling
                let cursor = std::io::Cursor::new(&mmap);
                let decoder_result = Decoder::new(cursor);
                
                let mut decoder = match decoder_result {
                    Ok(d) => d,
                    Err(e) => return Err(anyhow!("Failed to create decoder for {:?}: {}", path, e)),
                };
                
                // Use unlimited limits for large TIFFs
                decoder = decoder.with_limits(Limits::unlimited());
                
                let (width, height) = match decoder.dimensions() {
                    Ok(dims) => dims,
                    Err(e) => return Err(anyhow!("Failed to read dimensions from {:?}: {}", path, e)),
                };
                
                let image_result = decoder.read_image();
                
                let data = match image_result {
                    Ok(DecodingResult::U16(v)) => v,
                    Ok(DecodingResult::U8(v)) => v.into_iter().map(|x| x as u16).collect(),
                    Ok(DecodingResult::I16(v)) => v.into_iter().map(|x| x as u16).collect(),
                    Ok(other) => return Err(anyhow!("Unsupported decoding result for {:?}: {:?}", path, other)),
                    Err(e) => return Err(anyhow!("Failed to decode image from {:?}: {}", path, e)),
                };
                
                // Verify data length
                let expected_len = height as usize * width as usize;
                if data.len() != expected_len {
                    return Err(anyhow!(
                        "Data length mismatch in {:?}: got {}, expected {}",
                        path, data.len(), expected_len
                    ));
                }
                
                // Create array from data
                Array2::from_shape_vec((height as usize, width as usize), data)
                    .map_err(|e| anyhow!("Failed to create array from TIFF data for {:?}: {}", path, e))
            }));
            
            // Handle inner decode panic catch result
            match decode_result {
                Ok(result) => result,
                Err(_) => Err(anyhow!("Decoder panic while processing {:?}", path)),
            }
        }));
        
        // Handle middle panic catch result
        match inner_result {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Inner panic while processing {:?}", path)),
        }
    }));
    
    // Handle outer panic catch result
    match result {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Outer panic while reading TIFF from {:?}", path)),
    }
}

/// Process a band by upsampling low-resolution bands to full resolution
/// and then downsampling to the target resolution - optimized version
fn process_band(
    data: &Array2<u16>, 
    full_height: usize, 
    full_width: usize, 
    sample_rate: usize,
    expected_resolution: BandResolution,
    band_name: &str,
) -> Array2<u16> {
    let input_h = data.dim().0;
    let input_w = data.dim().1;
    let output_h = full_height / sample_rate;
    let output_w = full_width / sample_rate;
    let is_full = input_h == full_height && input_w == full_width;
    let is_half = input_h * 2 == full_height && input_w * 2 == full_width;

    match expected_resolution {
        BandResolution::Full if !is_full => {
            warn!(
                "Band {} expected 10m but got {}x{}, falling back to dimension-based processing",
                band_name, input_h, input_w
            );
        }
        BandResolution::Half if !is_half => {
            warn!(
                "Band {} expected 20m but got {}x{}, falling back to dimension-based processing",
                band_name, input_h, input_w
            );
        }
        _ => {}
    }
    
    // Optimized processing path based on dimensions
    if is_full {
        // Already full resolution, just downsample
        if sample_rate == 1 {
            // No downsampling needed, return a copy
            return data.clone();
        } else {
            return downsample_band_optimized(data, sample_rate, output_h, output_w);
        }
    } else {
        // Low-resolution band, need to upsample then downsample
        if sample_rate == 1 {
            // Only upsample needed
            return upsample_band_optimized(data, full_height, full_width);
        } else {
            // First upsample to full resolution
            let upsampled = upsample_band_optimized(data, full_height, full_width);
            
            // Then downsample to target resolution
            return downsample_band_optimized(&upsampled, sample_rate, output_h, output_w);
        }
    }
}

/// Optimized upsampling using pre-computed ratios and direct indexing
fn upsample_band_optimized(data: &Array2<u16>, full_height: usize, full_width: usize) -> Array2<u16> {
    let input_h = data.dim().0;
    let input_w = data.dim().1;
    
    // If already at target resolution, just return a copy
    if input_h == full_height && input_w == full_width {
        return data.clone();
    }
    
    // Calculate scaling factors once
    let h_ratio = (input_h as f64 - 1.0) / (full_height as f64 - 1.0);
    let w_ratio = (input_w as f64 - 1.0) / (full_width as f64 - 1.0);
    
    let mut upsampled = Array2::zeros((full_height, full_width));
    
    // Pre-calculate source y positions for each output y
    let src_y_positions: Vec<(usize, usize, f64)> = (0..full_height)
        .map(|y| {
            let src_y = (y as f64) * h_ratio;
            let src_y_floor = src_y.floor() as usize;
            let src_y_ceil = (src_y_floor + 1).min(input_h - 1);
            let y_lerp = src_y - src_y_floor as f64;
            (src_y_floor, src_y_ceil, y_lerp)
        })
        .collect();
    
    // Pre-calculate source x positions for each output x
    let src_x_positions: Vec<(usize, usize, f64)> = (0..full_width)
        .map(|x| {
            let src_x = (x as f64) * w_ratio;
            let src_x_floor = src_x.floor() as usize;
            let src_x_ceil = (src_x_floor + 1).min(input_w - 1);
            let x_lerp = src_x - src_x_floor as f64;
            (src_x_floor, src_x_ceil, x_lerp)
        })
        .collect();
    
    // Process sequentially to avoid borrow issues
    for y in 0..full_height {
        let (src_y_floor, src_y_ceil, y_lerp) = src_y_positions[y];
        
        for x in 0..full_width {
            let (src_x_floor, src_x_ceil, x_lerp) = src_x_positions[x];
            
            // Bilinear interpolation
            let top_left = data[[src_y_floor, src_x_floor]] as f64;
            let top_right = data[[src_y_floor, src_x_ceil]] as f64;
            let bottom_left = data[[src_y_ceil, src_x_floor]] as f64;
            let bottom_right = data[[src_y_ceil, src_x_ceil]] as f64;
            
            let top = top_left * (1.0 - x_lerp) + top_right * x_lerp;
            let bottom = bottom_left * (1.0 - x_lerp) + bottom_right * x_lerp;
            
            let value = top * (1.0 - y_lerp) + bottom * y_lerp;
            upsampled[[y, x]] = value.round() as u16;
        }
    }
    
    upsampled
}

/// Optimized downsampling with direct indices calculation
fn downsample_band_optimized(data: &Array2<u16>, sample_rate: usize, output_h: usize, output_w: usize) -> Array2<u16> {
    // Fast path for no downsampling
    if sample_rate == 1 {
        return data.clone();
    }
    
    let mut output = Array2::zeros((output_h, output_w));
    
    // Process sequentially since we need to modify the array
    for y in 0..output_h {
        let src_y = y * sample_rate;
        for x in 0..output_w {
            let src_x = x * sample_rate;
            output[[y, x]] = data[[src_y, src_x]];
        }
    }
    
    output
}

/// Resample a mask to specified dimensions
fn resample_mask(arr: &Array2<u8>, new_h: usize, new_w: usize) -> Array2<u8> {
    let (h, w) = arr.dim();
    let mut out = Array2::zeros((new_h, new_w));
    
    // Process sequentially since we need to modify the array
    for i in 0..new_h {
        // Calculate source pixel indices
        let src_i = (i * h) / new_h;
        for j in 0..new_w {
            let src_j = (j * w) / new_w;
            out[[i, j]] = arr[[src_i, src_j]];
        }
    }
    
    out
}

/// Downsample a mask to target resolution
fn downsample_mask(arr: &Array2<u8>, full_height: usize, full_width: usize, sample_rate: usize) -> Array2<u8> {
    let out_height = full_height / sample_rate;
    let out_width = full_width / sample_rate;
    let mut out = Array2::zeros((out_height, out_width));
    
    // Process sequentially since we need to modify the array
    for i in 0..out_height {
        let src_i = i * sample_rate;
        for j in 0..out_width {
            let src_j = j * sample_rate;
            out[[i, j]] = arr[[src_i, src_j]];
        }
    }
    
    out
}

/// Read TIFF dimensions safely without decoding full data
fn read_tiff_dimensions_safe(path: &Path) -> Result<(usize, usize)> {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, usize)> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(anyhow!("Failed to open {:?}: {}", path, e)),
        };

        let mmap = match unsafe { MmapOptions::new().map(&file) } {
            Ok(m) => m,
            Err(e) => return Err(anyhow!("Failed to mmap {:?}: {}", path, e)),
        };

        let decode_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, usize)> {
            let cursor = std::io::Cursor::new(&mmap);
            let decoder_result = Decoder::new(cursor);
            let mut decoder = match decoder_result {
                Ok(d) => d,
                Err(e) => return Err(anyhow!("Failed to create decoder for {:?}: {}", path, e)),
            };

            decoder = decoder.with_limits(Limits::unlimited());
            let (width, height) = decoder
                .dimensions()
                .map_err(|e| anyhow!("Failed to read dimensions from {:?}: {}", path, e))?;
            Ok((height as usize, width as usize))
        }));

        match decode_result {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Decoder panic while reading dimensions from {:?}", path)),
        }
    }));

    match result {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Outer panic while reading dimensions from {:?}", path)),
    }
}

/// Detect per-band resolution by reading the first available TIFF for each band
fn detect_band_resolutions(input_dir: &Path, full_height: usize, full_width: usize) -> Result<Vec<BandResolution>> {
    let mut resolutions = Vec::with_capacity(BANDS.len());

    for band_name in BANDS.iter() {
        let band_dir = input_dir.join(band_name);
        let pattern = band_dir.join("*.tiff").to_string_lossy().to_string();
        let mut dims = None;

        for entry in glob(&pattern)? {
            if let Ok(path) = entry {
                match read_tiff_dimensions_safe(&path) {
                    Ok((h, w)) => {
                        dims = Some((h, w));
                        break;
                    }
                    Err(e) => {
                        warn!("Failed to read dimensions for {:?}: {}", path, e);
                    }
                }
            }
        }

        let (h, w) = dims.ok_or_else(|| {
            anyhow!(
                "No valid TIFF files found for band {} in {:?}",
                band_name,
                band_dir
            )
        })?;

        let resolution = if h == full_height && w == full_width {
            BandResolution::Full
        } else if h * 2 == full_height && w * 2 == full_width {
            BandResolution::Half
        } else {
            return Err(anyhow!(
                "Unexpected resolution for band {}: {}x{} (full is {}x{})",
                band_name,
                h,
                w,
                full_height,
                full_width
            ));
        };

        resolutions.push(resolution);
    }

    Ok(resolutions)
}

/// Get full resolution dimensions from the first valid TIFF file in the "red" directory
fn get_full_resolution(input_dir: &Path) -> Result<(usize, usize)> {
    let red_dir = input_dir.join("red");
    let pattern = red_dir.join("*.tiff").to_string_lossy().to_string();
    info!("Looking for sample TIFF with pattern: {}", pattern);
    
    for entry in glob(&pattern)? {
        if let Ok(path) = entry {
            match read_tiff_safe(&path) {
                Ok(arr) => {
                    let (h, w) = arr.dim();
                    return Ok((h, w));
                },
                Err(e) => {
                    warn!("Error reading sample TIFF {:?}: {}", path, e);
                    continue;
                }
            }
        }
    }
    Err(anyhow!("No valid TIFF file found in the 'red' folder at {:?}", red_dir))
}

/* ============= NPY file helper functions ============= */

/// Define trait for NPY data types
trait NpyDtype {
    fn npy_dtype() -> &'static str;
}
impl NpyDtype for u8 {
    fn npy_dtype() -> &'static str { "|u1" }
}
impl NpyDtype for u16 {
    fn npy_dtype() -> &'static str { "<u2" }
}
impl NpyDtype for f64 {
    fn npy_dtype() -> &'static str { "<f8" }
}

/// Create a new NPY file with header and return a buffered writer
fn create_npy_file<T: NpyDtype, const N: usize>(
    path: &Path,
    shape: &[usize; N],
) -> Result<BufWriter<File>> {
    // Open file with large buffer (512MB)
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(512 * 1024 * 1024, file);

    // Write numpy magic
    writer.write_all(b"\x93NUMPY")?;
    // Version 1.0
    writer.write_all(&[1, 0])?;

    // Construct metadata dictionary
    let dtype = T::npy_dtype();
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype, shape_str
    );

    // Align to 16 bytes (including newline)
    let header_len = header_dict.len() + 1;
    let pad = if header_len % 16 == 0 { 0 } else { 16 - (header_len % 16) };
    let padded_header = format!("{}{}", header_dict, " ".repeat(pad));
    let final_header = format!("{}\n", padded_header);
    let header_bytes = final_header.as_bytes();

    // Write header length as 2-byte little-endian
    let header_len_u16 = header_bytes.len() as u16;
    writer.write_all(&header_len_u16.to_le_bytes())?;
    writer.write_all(header_bytes)?;

    // Return the writer positioned at the start of the data section
    Ok(writer)
}

/// Write a small numpy array in one operation
fn write_npy_simple<T: Copy + NpyDtype, const N: usize>(
    path: &Path,
    shape: &[usize; N],
    data: &[T],
) -> Result<()> {
    let expected_len: usize = shape.iter().product();
    if data.len() != expected_len {
        return Err(anyhow!(
            "Data length mismatch: got {}, expected {} for shape {:?}",
            data.len(),
            expected_len,
            shape
        ));
    }

    // Use BufWriter with smaller buffer for small arrays
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    // Write magic
    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])?;

    // Construct metadata
    let dtype = T::npy_dtype();
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_str = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        dtype, shape_str
    );

    let header_len = header_dict.len() + 1;
    let pad = if header_len % 16 == 0 { 0 } else { 16 - (header_len % 16) };
    let padded_header = format!("{}{}", header_dict, " ".repeat(pad));
    let final_header = format!("{}\n", padded_header);
    let header_bytes = final_header.as_bytes();
    let header_len_u16 = header_bytes.len() as u16;
    writer.write_all(&header_len_u16.to_le_bytes())?;
    writer.write_all(header_bytes)?;

    // Write data
    let data_bytes = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };
    writer.write_all(data_bytes)?;
    writer.flush()?;

    Ok(())
}