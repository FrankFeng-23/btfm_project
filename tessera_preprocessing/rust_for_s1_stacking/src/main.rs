// src/main.rs
use anyhow::{anyhow, Result};
use chrono::{NaiveDate, Datelike};
use env_logger::{Builder, Env};
use glob::glob;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{info, warn, error};
use memmap2::MmapOptions;
use ndarray::Array3;
use rayon::ThreadPoolBuilder;
use regex::Regex;
use std::panic::AssertUnwindSafe;
use std::{
    collections::HashMap,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use structopt::StructOpt;
use tiff::decoder::{Decoder, DecodingResult, Limits};

#[derive(Debug, StructOpt)]
#[structopt(name = "s1_stack", about = "Process Sentinel-1 data for a single tile")]
struct Opt {
    /// Input directory (where TIFF files are)
    #[structopt(short = "i", long = "input-dir", parse(from_os_str))]
    input_dir: PathBuf,

    /// Output directory (where processed NPY files will go)
    #[structopt(short = "o", long = "output-dir", parse(from_os_str))]
    output_dir: PathBuf,
    
    /// Number of parallel processes to use
    #[structopt(short = "p", long = "parallel", default_value = "8")]
    parallel: usize,
    
    /// Downsampling rate (e.g., 10 means take every 10th pixel)
    #[structopt(short = "r", long = "rate", default_value = "10")]
    rate: usize,
}

// Structure to hold metadata about a TIFF file
#[derive(Debug, Clone)]
struct TiffMetadata {
    path: PathBuf,
    date: NaiveDate,
    orbit: String,         // "ascending" or "descending"
    polarization: String,  // "vv" or "vh"
    doy: u16,              // Day of year (1-366)
}

// Structure to hold a pair of VV/VH files for a date
#[derive(Debug, Clone)]
struct PolarizationPair {
    date: NaiveDate,
    doy: u16,
    vv_path: Option<PathBuf>,
    vh_path: Option<PathBuf>,
}

// Structure to hold successfully processed data
#[derive(Debug, Clone)]
struct ProcessedPair {
    date: NaiveDate,
    doy: u16,
    data: Vec<i16>,  // flattened array of shape [height, width, 2]
}

fn main() -> Result<()> {
    // Initialize env_logger
    let mut builder = Builder::from_env(Env::default().default_filter_or("info"));
    builder.format_timestamp_secs();
    builder.init();

    let opt = Opt::from_args();

    info!("Starting s1_stack with {} parallel processes...", opt.parallel);
    info!("Input dir: {:?}", opt.input_dir);
    info!("Output dir: {:?}", opt.output_dir);
    info!("Downsampling rate: {}", opt.rate);

    // Set up Rayon thread pool with specified number of threads
    ThreadPoolBuilder::new()
        .num_threads(opt.parallel)
        .build_global()?;
    
    info!("Initialized thread pool with {} threads", opt.parallel);

    // Create output directory if it doesn't exist
    fs::create_dir_all(&opt.output_dir)?;

    // Find all TIFF files and extract metadata
    info!("Searching for TIFF files in input directory...");
    let tiff_files = find_tiff_files(&opt.input_dir)?;
    info!("Found {} TIFF files", tiff_files.len());

    // Get image dimensions - either from first file or use defaults if no files
    let (height, width) = if tiff_files.is_empty() {
        // Default dimensions if no files found
        let default_h = 868 / opt.rate;
        let default_w = 782 / opt.rate;
        warn!("No TIFF files found. Using default downsampled dimensions: H={}, W={}", default_h, default_w);
        (default_h, default_w)
    } else {
        // Try to get dimensions from first few files until we find a valid one
        let mut dimensions = None;
        for file in &tiff_files[..tiff_files.len().min(10)] {
            match get_tiff_dimensions_downsampled(&file.path, opt.rate) {
                Ok((h, w)) => {
                    info!("Got dimensions from {:?}", file.path);
                    info!("Original TIFF dimensions before downsampling: height={}, width={}", h * opt.rate, w * opt.rate);
                    info!("Downsampled TIFF dimensions: height={}, width={}, bands=1", h, w);
                    dimensions = Some((h, w));
                    break;
                }
                Err(e) => {
                    warn!("Failed to get dimensions from {:?}: {}", file.path, e);
                }
            }
        }
        
        match dimensions {
            Some(dims) => dims,
            None => {
                return Err(anyhow!("Could not get dimensions from any TIFF file"));
            }
        }
    };

    // Group files by orbit and pair VV/VH by date
    let (ascending_pairs, descending_pairs) = group_and_pair_files(tiff_files);
    
    info!("Grouped files: {} ascending dates, {} descending dates", 
          ascending_pairs.len(), descending_pairs.len());

    // Set up multi-progress
    let mp = MultiProgress::new();

    // Process both orbits in parallel if they both exist
    let mut handles = vec![];
    
    if !ascending_pairs.is_empty() {
        info!("Processing {} ascending orbit dates...", ascending_pairs.len());
        let output_dir = opt.output_dir.clone();
        let pairs_clone = ascending_pairs.clone();
        let mp_clone = mp.clone();
        let rate = opt.rate;
        
        let handle = std::thread::spawn(move || {
            match process_orbit_pairs(&pairs_clone, &output_dir, height, width, "ascending", mp_clone, rate) {
                Ok(_) => info!("Completed processing ascending orbit files"),
                Err(e) => error!("Error processing ascending orbit files: {}", e),
            }
        });
        handles.push(handle);
    } else {
        warn!("No ascending orbit files found. Creating empty output files.");
        create_empty_orbit_files(&opt.output_dir, height, width, "ascending")?;
    }

    if !descending_pairs.is_empty() {
        info!("Processing {} descending orbit dates...", descending_pairs.len());
        let output_dir = opt.output_dir.clone();
        let pairs_clone = descending_pairs.clone();
        let mp_clone = mp.clone();
        let rate = opt.rate;
        
        let handle = std::thread::spawn(move || {
            match process_orbit_pairs(&pairs_clone, &output_dir, height, width, "descending", mp_clone, rate) {
                Ok(_) => info!("Completed processing descending orbit files"),
                Err(e) => error!("Error processing descending orbit files: {}", e),
            }
        });
        handles.push(handle);
    } else {
        warn!("No descending orbit files found. Creating empty output files.");
        create_empty_orbit_files(&opt.output_dir, height, width, "descending")?;
    }

    // Wait for all threads to complete
    for handle in handles {
        match handle.join() {
            Ok(_) => {},
            Err(e) => error!("Thread panic: {:?}", e),
        }
    }
    
    // Clear the progress bars when done
    mp.clear()?;

    info!("All done! Output in {:?}", opt.output_dir);
    Ok(())
}

/// Find all TIFF files in the directory and extract metadata
fn find_tiff_files(dir: &Path) -> Result<Vec<TiffMetadata>> {
    let pattern = dir.join("*.tiff").to_string_lossy().to_string();
    info!("Searching for TIFF files with pattern: {}", pattern);
    
    // Regular expression to extract date, polarization, and orbit from filename
    let re = Regex::new(r"(\d{4}-\d{2}-\d{2})_(vv|vh)_(ascending|descending)\.tiff$")?;
    
    let entries: Vec<_> = glob(&pattern)?.collect::<Result<Vec<_>, _>>()?;
    info!("Found {} potential TIFF files", entries.len());
    
    // Process files in parallel
    let metadata_mutex = Arc::new(Mutex::new(Vec::new()));
    
    rayon::scope(|s| {
        for path in entries {
            let metadata_mutex = Arc::clone(&metadata_mutex);
            let re = re.clone();
            
            s.spawn(move |_| {
                let filename = match path.file_name().and_then(|n| n.to_str()) {
                    Some(name) => name,
                    None => {
                        warn!("Invalid filename: {:?}", path);
                        return;
                    }
                };
                
                // Extract date, polarization, and orbit from filename
                if let Some(caps) = re.captures(filename) {
                    let date_str = &caps[1];
                    let polarization = caps[2].to_string();
                    let orbit = caps[3].to_string();
                    
                    match NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                        Ok(date) => {
                            let doy = date.ordinal() as u16;
                            let meta = TiffMetadata {
                                path: path.clone(),
                                date,
                                orbit: orbit.clone(),
                                polarization: polarization.clone(),
                                doy,
                            };
                            
                            let mut metadata = metadata_mutex.lock().unwrap();
                            metadata.push(meta);
                            
                            if metadata.len() % 100 == 0 {
                                info!("Processed {} files so far", metadata.len());
                            }
                        },
                        Err(e) => {
                            warn!("Failed to parse date from {}: {}", filename, e);
                        }
                    }
                } else {
                    warn!("Filename does not match expected pattern: {}", filename);
                }
            });
        }
    });
    
    // Get the final result
    let mut metadata = Arc::try_unwrap(metadata_mutex)
        .expect("Failed to unwrap mutex")
        .into_inner()?;
    
    // Sort metadata by date
    metadata.sort_by(|a, b| a.date.cmp(&b.date));
    
    info!("Successfully processed {} valid TIFF files", metadata.len());
    Ok(metadata)
}

/// Group files by orbit and pair VV/VH files by date
fn group_and_pair_files(files: Vec<TiffMetadata>) -> (Vec<PolarizationPair>, Vec<PolarizationPair>) {
    // Group files by orbit and date
    let mut ascending_map: HashMap<NaiveDate, (Option<PathBuf>, Option<PathBuf>)> = HashMap::new();
    let mut descending_map: HashMap<NaiveDate, (Option<PathBuf>, Option<PathBuf>)> = HashMap::new();
    
    // Fill the maps with files
    for file in files {
        let map = if file.orbit == "ascending" { &mut ascending_map } else { &mut descending_map };
        
        let entry = map.entry(file.date).or_insert((None, None));
        
        if file.polarization == "vv" {
            entry.0 = Some(file.path);
        } else if file.polarization == "vh" {
            entry.1 = Some(file.path);
        }
    }
    
    // Convert maps to vectors of pairs
    let mut ascending_pairs: Vec<PolarizationPair> = ascending_map
        .into_iter()
        .map(|(date, (vv_path, vh_path))| PolarizationPair {
            date,
            doy: date.ordinal() as u16,
            vv_path,
            vh_path,
        })
        .collect();
    
    let mut descending_pairs: Vec<PolarizationPair> = descending_map
        .into_iter()
        .map(|(date, (vv_path, vh_path))| PolarizationPair {
            date,
            doy: date.ordinal() as u16,
            vv_path,
            vh_path,
        })
        .collect();
    
    // Sort pairs by date
    ascending_pairs.sort_by(|a, b| a.date.cmp(&b.date));
    descending_pairs.sort_by(|a, b| a.date.cmp(&b.date));
    
    (ascending_pairs, descending_pairs)
}

/// Safe wrapper for processing a single polarization file
fn process_single_file_safe(path: &Path, rate: usize) -> Result<Array3<i16>> {
    // Triple-wrapped panic catching for maximum safety
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array3<i16>> {
        let inner_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array3<i16>> {
            read_tiff_mmap_downsampled(path, rate)
        }));
        
        match inner_result {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Inner panic while processing {:?}", path)),
        }
    }));
    
    match result {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Outer panic while processing {:?}", path)),
    }
}

/// Process a group of polarization pairs for a specific orbit (ascending or descending)
fn process_orbit_pairs(
    pairs: &[PolarizationPair], 
    output_dir: &Path, 
    height: usize, 
    width: usize, 
    orbit: &str,
    mp: MultiProgress,
    rate: usize,
) -> Result<()> {
    let num_pairs = pairs.len();
    let num_bands = 2;
    
    // Create progress bar
    let style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("#>-");
    let pb = mp.add(ProgressBar::new(num_pairs as u64));
    pb.set_style(style);
    pb.set_message(format!("Processing {} orbit files...", orbit));
    
    // Collect successfully processed data
    let processed_data_mutex = Arc::new(Mutex::new(Vec::new()));
    
    // Process pairs in parallel chunks
    const CHUNK_SIZE: usize = 5;
    
    let pairs_chunks: Vec<_> = pairs.chunks(CHUNK_SIZE).collect();
    let total_chunks = pairs_chunks.len();
    info!("Split processing into {} chunks", total_chunks);
    
    // Process chunks in parallel
    rayon::scope(|s| {
        for (chunk_idx, chunk) in pairs_chunks.iter().enumerate() {
            let processed_data_mutex = Arc::clone(&processed_data_mutex);
            let pb = pb.clone();
            
            s.spawn(move |_| {
                info!("Processing chunk {}/{} with {} pairs", chunk_idx + 1, total_chunks, chunk.len());
                
                // Process each pair in the chunk
                for pair in *chunk {
                    // Wrap entire pair processing in panic catching
                    let pair_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<ProcessedPair> {
                        // Create buffer for this pair
                        let mut combined_data = vec![0i16; height * width * num_bands];
                        let mut pair_valid = true;
                        
                        // Process VV (first band)
                        if let Some(vv_path) = &pair.vv_path {
                            match process_single_file_safe(vv_path, rate) {
                                Ok(vv_data) => {
                                    // Copy VV data (band 0) to combined buffer (band 0)
                                    for i in 0..height {
                                        for j in 0..width {
                                            combined_data[i * width * num_bands + j * num_bands] = vv_data[[0, i, j]];
                                        }
                                    }
                                },
                                Err(e) => {
                                    error!("Error reading VV band from {:?}: {}", vv_path, e);
                                    pair_valid = false;
                                }
                            }
                        } else {
                            warn!("Missing VV band for date {}, using zeros", pair.date);
                        }
                        
                        // Process VH (second band)
                        if let Some(vh_path) = &pair.vh_path {
                            match process_single_file_safe(vh_path, rate) {
                                Ok(vh_data) => {
                                    // Copy VH data (band 0) to combined buffer (band 1)
                                    for i in 0..height {
                                        for j in 0..width {
                                            combined_data[i * width * num_bands + j * num_bands + 1] = vh_data[[0, i, j]];
                                        }
                                    }
                                },
                                Err(e) => {
                                    error!("Error reading VH band from {:?}: {}", vh_path, e);
                                    pair_valid = false;
                                }
                            }
                        } else {
                            warn!("Missing VH band for date {}, using zeros", pair.date);
                        }
                        
                        // Return result based on validity
                        if pair_valid {
                            Ok(ProcessedPair {
                                date: pair.date,
                                doy: pair.doy,
                                data: combined_data,
                            })
                        } else {
                            Err(anyhow!("Pair processing failed for date {}", pair.date))
                        }
                    }));
                    
                    // Handle the pair processing result
                    match pair_result {
                        Ok(Ok(processed)) => {
                            let mut processed_data = processed_data_mutex.lock().unwrap();
                            processed_data.push(processed);
                        },
                        Ok(Err(e)) => {
                            error!("Error processing pair for date {}: {}", pair.date, e);
                        },
                        Err(_) => {
                            error!("Panic occurred while processing pair for date {}. Skipping this timestep.", pair.date);
                        }
                    }
                    
                    // Update progress
                    pb.inc(1);
                }
            });
        }
    });
    
    pb.finish_with_message(format!("Processed all {} orbit files", orbit));
    
    // Get the processed data
    let mut processed_data = Arc::try_unwrap(processed_data_mutex)
        .expect("Failed to unwrap processed data mutex")
        .into_inner()?;
    
    // Sort strictly by full date to ensure year-aware ordering (previous year first)
    processed_data.sort_by(|a, b| a.date.cmp(&b.date));
    
    let num_successful = processed_data.len();
    info!("Successfully processed {} out of {} pairs for {} orbit", num_successful, num_pairs, orbit);
    
    if num_successful == 0 {
        warn!("No successful pairs for {} orbit. Creating empty output files.", orbit);
        create_empty_orbit_files(output_dir, height, width, orbit)?;
        return Ok(());
    }
    
    // Write the data to NPY file
    let data_file_path = output_dir.join(format!("sar_{}.npy", orbit));
    let mut data_file = create_npy_file::<i16, 4>(
        &data_file_path,
        &[num_successful, height, width, num_bands],
    )?;
    
    // Write all the data
    for processed in &processed_data {
        let slice_bytes = unsafe {
            std::slice::from_raw_parts(
                processed.data.as_ptr() as *const u8,
                processed.data.len() * std::mem::size_of::<i16>(),
            )
        };
        data_file.write_all(slice_bytes)?;
    }
    
    data_file.flush()?;
    info!("Created data file: {:?}", data_file_path);
    
    // Write DOY file
    let doys: Vec<i16> = processed_data.iter().map(|p| p.doy as i16).collect();
    let doy_file_path = output_dir.join(format!("sar_{}_doy.npy", orbit));
    write_npy_simple::<i16, 1>(&doy_file_path, &[doys.len()], &doys)?;
    info!("Created DOY file: {:?}", doy_file_path);
    
    Ok(())
}

/// Create empty output files for an orbit type
fn create_empty_orbit_files(output_dir: &Path, height: usize, width: usize, orbit: &str) -> Result<()> {
    // Create empty data file
    let data_file_path = output_dir.join(format!("sar_{}.npy", orbit));
    let mut data_file = create_npy_file::<i16, 4>(
        &data_file_path,
        &[0, height, width, 2],
    )?;
    data_file.flush()?;
    info!("Created empty {} data file: {:?}", orbit, data_file_path);
    
    // Create empty DOY file
    let doy_file_path = output_dir.join(format!("sar_{}_doy.npy", orbit));
    let empty_doys: Vec<i16> = Vec::new();
    write_npy_simple::<i16, 1>(
        &doy_file_path,
        &[0],
        &empty_doys,
    )?;
    info!("Created empty {} DOY file: {:?}", orbit, doy_file_path);
    
    Ok(())
}

/// Get the dimensions of a TIFF file after downsampling
fn get_tiff_dimensions_downsampled(path: &Path, rate: usize) -> Result<(usize, usize)> {
    info!("Reading dimensions from: {:?}", path);
    
    // Triple-wrapped panic catching for maximum safety
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, usize)> {
        let inner_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, usize)> {
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => return Err(anyhow!("Failed to open {:?}: {}", path, e)),
            };
            
            let mmap = match unsafe { MmapOptions::new().map(&file) } {
                Ok(m) => m,
                Err(e) => return Err(anyhow!("Failed to mmap {:?}: {}", path, e)),
            };
            
            // Wrap decoder operations in another catch_unwind
            let decode_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(usize, usize)> {
                let cursor = std::io::Cursor::new(&mmap);
                let decoder_result = Decoder::new(cursor);
                
                let mut decoder = match decoder_result {
                    Ok(d) => d,
                    Err(e) => return Err(anyhow!("Failed to create decoder for {:?}: {}", path, e)),
                };
                
                // Use unlimited limits
                decoder = decoder.with_limits(Limits::unlimited());
                
                match decoder.dimensions() {
                    Ok((width, height)) => {
                        // Calculate downsampled dimensions
                        let ds_height = height as usize / rate;
                        let ds_width = width as usize / rate;
                        Ok((ds_height, ds_width))
                    },
                    Err(e) => Err(anyhow!("Failed to get dimensions from {:?}: {}", path, e)),
                }
            }));
            
            match decode_result {
                Ok(result) => result,
                Err(_) => Err(anyhow!("Decoder panic while getting dimensions from {:?}", path)),
            }
        }));
        
        match inner_result {
            Ok(result) => result,
            Err(_) => Err(anyhow!("Inner panic while getting dimensions from {:?}", path)),
        }
    }));
    
    match result {
        Ok(result) => result,
        Err(_) => Err(anyhow!("Outer panic while reading TIFF dimensions from {:?}", path)),
    }
}

/// Read a TIFF file using memory mapping and return a downsampled ndarray
/// Now each file contains only one band of data
fn read_tiff_mmap_downsampled(path: &Path, rate: usize) -> Result<Array3<i16>> {
    // Triple-wrapped panic catching to handle all possible panics
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array3<i16>> {
        let inner_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array3<i16>> {
            // Try to open file with timeout
            let file = match File::open(path) {
                Ok(f) => f,
                Err(e) => return Err(anyhow!("Failed to open {:?}: {}", path, e)),
            };
            
            // Memory map with error handling
            let mmap = match unsafe { MmapOptions::new().map(&file) } {
                Ok(m) => m,
                Err(e) => return Err(anyhow!("Failed to mmap {:?}: {}", path, e)),
            };
            
            // Wrap decoder operations in another catch_unwind for extra safety
            let decode_result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<Array3<i16>> {
                // Create decoder with explicit error handling
                let cursor = std::io::Cursor::new(&mmap);
                let decoder_result = Decoder::new(cursor);
                
                let mut decoder = match decoder_result {
                    Ok(d) => d,
                    Err(e) => return Err(anyhow!("Failed to create decoder for {:?}: {}", path, e)),
                };
                
                // Use unlimited limits
                decoder = decoder.with_limits(Limits::unlimited());
                
                // Get dimensions with error handling
                let (width, height) = match decoder.dimensions() {
                    Ok(dims) => dims,
                    Err(e) => return Err(anyhow!("Failed to get dimensions from {:?}: {}", path, e)),
                };
                
                // Calculate downsampled dimensions
                let ds_height = height as usize / rate;
                let ds_width = width as usize / rate;
                
                // Read image data with detailed error handling
                let image_result = decoder.read_image();
                
                match image_result {
                    Ok(DecodingResult::I16(data)) => {
                        // Expected data length for 1 band
                        let expected_len = height as usize * width as usize;
                        if data.len() != expected_len {
                            return Err(anyhow!(
                                "Data length mismatch in {:?}: got {}, expected {} for shape (1, {}, {})",
                                path, data.len(), expected_len, height, width
                            ));
                        }
                        
                        // Create a new array with downsampled dimensions
                        let mut arr = Array3::<i16>::zeros((1, ds_height, ds_width));
                        
                        // Perform downsampling by taking every rate-th pixel
                        for i in 0..ds_height {
                            for j in 0..ds_width {
                                let src_i = i * rate;
                                let src_j = j * rate;
                                let idx = src_i * width as usize + src_j;
                                if idx < data.len() {
                                    arr[[0, i, j]] = data[idx];
                                }
                            }
                        }
                        
                        Ok(arr)
                    },
                    Ok(DecodingResult::U16(data)) => {
                        // Convert u16 to i16 if necessary
                        let i16_data: Vec<i16> = data.into_iter().map(|x| x as i16).collect();
                        let expected_len = height as usize * width as usize;
                        if i16_data.len() != expected_len {
                            return Err(anyhow!(
                                "Data length mismatch in {:?}: got {}, expected {} for shape (1, {}, {})",
                                path, i16_data.len(), expected_len, height, width
                            ));
                        }
                        
                        // Create a new array with downsampled dimensions
                        let mut arr = Array3::<i16>::zeros((1, ds_height, ds_width));
                        
                        // Perform downsampling by taking every rate-th pixel
                        for i in 0..ds_height {
                            for j in 0..ds_width {
                                let src_i = i * rate;
                                let src_j = j * rate;
                                let idx = src_i * width as usize + src_j;
                                if idx < i16_data.len() {
                                    arr[[0, i, j]] = i16_data[idx];
                                }
                            }
                        }
                        
                        Ok(arr)
                    },
                    Ok(other) => Err(anyhow!("Unsupported decoding result in {:?}: {:?}", path, other)),
                    Err(e) => Err(anyhow!("Failed to decode image from {:?}: {}", path, e)),
                }
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

/* ============= NPY file writing helper functions ============= */

/// Trait for numpy dtype
trait NpyDtype {
    fn npy_dtype() -> &'static str;
}

impl NpyDtype for i16 {
    fn npy_dtype() -> &'static str { "<i2" }
}

/// Create a new empty NPY file with header and return a buffered writer
fn create_npy_file<T: NpyDtype, const N: usize>(
    path: &Path,
    shape: &[usize; N],
) -> Result<BufWriter<File>> {
    // Open file with 256MB buffer
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024 * 256, file);

    // Write numpy magic
    writer.write_all(b"\x93NUMPY")?;
    // Version 1.0
    writer.write_all(&[1, 0])?;

    // Create header dictionary
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

    // File pointer now at start of data, return writer
    Ok(writer)
}

/// Write a "small" numpy array all at once
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

    // Use BufWriter with 1MB buffer for small arrays
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    // Write magic
    writer.write_all(b"\x93NUMPY")?;
    writer.write_all(&[1, 0])?;

    // Create header
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