#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
s1_stack.py

Once .tif are created, stack them into sar_ascending.npy, etc.

Env vars:
    S1_DATA_RAW        (possibly if no _processed)
    S1_DATA_PROCESSED  (where .tif might be)
    S2_DATA_PROCESSED  (where final npy go)
"""

import argparse
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("org.esa.snap.core.util.ServiceLoader").setLevel(logging.ERROR)
logging.getLogger("javax.media.jai").setLevel(logging.ERROR)
import os
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from osgeo import gdal
import shutil

def remove_raw_s1(tile_id: str, data_sar_dir="data_s1_raw"):
        p = Path(data_sar_dir) / tile_id
        if p.exists():
                shutil.rmtree(p, ignore_errors=True)
                logging.info(f"[S1] Removed raw SAR data: {p}")

def amplitude_to_db_scaled(amp):
        amp_safe = np.maximum(amp, 1e-6)
        db = 20.0 * np.log10(amp_safe)
        db_shift = db + 50.0
        scaled = db_shift * 200.0
        return np.clip(scaled,0,30000).astype(np.int16)

def read_tiff_resample(tiff_path, ref_proj, ref_gt, out_size):
        ds_in = gdal.Open(tiff_path,gdal.GA_ReadOnly)
        if not ds_in:
                raise IOError(f"Cannot open {tiff_path}")
        mem_drv = gdal.GetDriverByName("MEM")
        ds_out = mem_drv.Create("", out_size[0], out_size[1], 2, gdal.GDT_Float32)
        ds_out.SetProjection(ref_proj)
        ds_out.SetGeoTransform(ref_gt)
        gdal.Warp(ds_out, ds_in, resampleAlg=gdal.GRA_NearestNeighbour)
        b1 = ds_out.GetRasterBand(1).ReadAsArray()
        b2 = ds_out.GetRasterBand(2).ReadAsArray()
        ds_in = None
        ds_out= None
        return np.stack([b1,b2],axis=0)

def mosaic_tiffs(tiff_paths, ref_proj, ref_gt, out_size):
        from osgeo import gdal
        mem = gdal.GetDriverByName("MEM")
        ds_out = mem.Create("", out_size[0], out_size[1], 2, gdal.GDT_Float32)
        ds_out.SetProjection(ref_proj)
        ds_out.SetGeoTransform(ref_gt)

        left = ref_gt[0]
        top  = ref_gt[3]
        px   = ref_gt[1]
        py   = ref_gt[5]
        right = left + px*out_size[0]
        bottom=top  + py*out_size[1]
        xRes=abs(px)
        yRes=abs(py)

        for tf in tiff_paths:
                ds_in = gdal.Open(str(tf), gdal.GA_ReadOnly)
                if not ds_in:
                        continue
                try:
                        gdal.Warp(ds_out, ds_in,
                                dstSRS=ref_proj,
                                outputBounds=(min(left,right), min(bottom,top), max(left,right), max(bottom,top)),
                                xRes=xRes, yRes=yRes,
                                srcNodata=0, dstNodata=0,
                                resampleAlg=gdal.GRA_NearestNeighbour
                        )
                except Exception as e:
                        logging.error(f"Cannot warp {tf}: {e}")
                ds_in = None

        b1 = ds_out.GetRasterBand(1).ReadAsArray()
        b2 = ds_out.GetRasterBand(2).ReadAsArray()
        ds_out = None
        return np.stack([b1,b2],axis=0)

def get_doy(fname):
        # Try to match the old format without orbit information
        m = re.search(r"(\d{8})T\d{6}", fname)
        if not m:
                # Try to match the new format with orbit information
                m = re.search(r"(\d{8})T\d{6}.*_(ASCENDING|DESCENDING)", fname)
                if not m:
                        return None
                # If the new format is matched, take the date part
                dt = datetime.strptime(m.group(1), "%Y%m%d")
        else:
                dt = datetime.strptime(m.group(1), "%Y%m%d")

        return dt.timetuple().tm_yday

def main():
        try:
                parser = argparse.ArgumentParser()
                parser.add_argument("--tile_id", required=True)
                parser.add_argument("--reference_tiff", required=True)
                parser.add_argument("--sample_rate", type=int, default=1)
                args = parser.parse_args()

                # read env
                S1_DATA_RAW       = os.environ.get("S1_DATA_RAW")
                S1_DATA_PROCESSED = os.environ.get("S1_DATA_PROCESSED")
                S2_DATA_PROCESSED = os.environ.get("S2_DATA_PROCESSED")

                ds_ref = gdal.Open(args.reference_tiff, gdal.GA_ReadOnly)
                if not ds_ref:
                        logging.error(f"Cannot open ref {args.reference_tiff}")
                        return
                xsize = ds_ref.RasterXSize
                ysize = ds_ref.RasterYSize
                ref_gt = ds_ref.GetGeoTransform()
                ref_proj = ds_ref.GetProjection()
                ds_ref = None

                out_w = max(1, xsize//args.sample_rate)
                out_h = max(1, ysize//args.sample_rate)

                # we look for .tif in S1_DATA_PROCESSED/<tile> or S1_DATA_RAW/<tile>
                tile_id = args.tile_id
                p_proc = Path(S1_DATA_PROCESSED) / tile_id
                p_raw  = Path(S1_DATA_RAW) / tile_id
                logging.info(f"Looking for S1 .tif in {p_proc}")

                tiff_list = []
                if p_proc.is_dir():
                        tiff_list = sorted(p_proc.glob("*.tif"))
                if not tiff_list and p_raw.is_dir():
                        tiff_list = sorted(p_raw.glob("*.tif"))

                if not tiff_list:
                        logging.warning(f"No S1 .tif for tile {tile_id}")
                        return
                logging.info(f"{len(tiff_list)} S1 tifs found for tile {tile_id}")

                # After finding all tiff files, classify them by orbit type
                asc_files = []
                desc_files = []
                for tf in tiff_list:
                        fname = tf.name.upper()
                        # Use the new naming convention to determine the orbit type
                        if "_DESCENDING" in fname:
                                desc_files.append(tf)
                        elif "_ASCENDING" in fname:
                                asc_files.append(tf)
                        else:
                                # Backward compatibility for old file naming format
                                if "DES" in fname:
                                        desc_files.append(tf)
                                else:
                                        asc_files.append(tf)

                logging.info(f"Found {len(asc_files)} ascending and {len(desc_files)} descending files")

                from collections import defaultdict
                asc_map = defaultdict(list)
                desc_map= defaultdict(list)

                for f in asc_files:
                        doy = get_doy(f.name)
                        if doy is None: doy=1
                        asc_map[doy].append(f)
                for f in desc_files:
                        doy = get_doy(f.name)
                        if doy is None: doy=1
                        desc_map[doy].append(f)

                def build_stack(dmap):
                        doys_sorted = sorted(dmap.keys())
                        T = len(doys_sorted)
                        arr_4d = np.zeros((T, out_h, out_w, 2), dtype=np.int16)
                        doys_arr= np.zeros((T,), dtype=np.int32)
                        for i, d in enumerate(doys_sorted):
                                doys_arr[i]=d
                                paths = dmap[d]
                                if len(paths)==1:
                                        arr2d = read_tiff_resample(str(paths[0]), ref_proj, ref_gt,(out_w,out_h))
                                else:
                                        arr2d = mosaic_tiffs(paths, ref_proj, ref_gt,(out_w,out_h))
                                vv = arr2d[0]
                                vh = arr2d[1]
                                arr_4d[i,:,:,0] = amplitude_to_db_scaled(vv)
                                arr_4d[i,:,:,1] = amplitude_to_db_scaled(vh)
                        return arr_4d, doys_arr

                sar_asc, asc_doys = build_stack(asc_map)
                sar_desc,desc_doys= build_stack(desc_map)

                # Add validation code to check DOY counts between original ZIPs and final stacks
                all_original_doys = set()
                original_zip_count = 0
                source_description = ""

                if p_proc.exists():
                        processed_tiffs = list(p_proc.glob("*.tif"))
                        original_zip_count = len(processed_tiffs)
                        for tf in processed_tiffs:
                                doy = get_doy(tf.name)
                                if doy is not None:
                                        all_original_doys.add(doy)

                        source_description = "TIFF files"

                # Count unique DOYs in stacked results
                stacked_unique_doys = set(asc_doys.tolist()).union(set(desc_doys.tolist()))

                # Log the comparison
                logging.info(f"[S1 validation] Found {original_zip_count} S1 files via {source_description}")
                logging.info(f"[S1 validation] Original unique DOYs: {len(all_original_doys)}, values: {sorted(all_original_doys)}")
                logging.info(f"[S1 validation] Stacked unique DOYs: {len(stacked_unique_doys)}, values: {sorted(stacked_unique_doys)}")

                # Add warning if counts don't match
                if len(all_original_doys) != len(stacked_unique_doys):
                        logging.warning(f"[S1 validation] DOY count mismatch! Original: {len(all_original_doys)}, Stacked: {len(stacked_unique_doys)}")
                        # Log specific differences
                        missing_doys = all_original_doys - stacked_unique_doys
                        extra_doys = stacked_unique_doys - all_original_doys
                        if missing_doys:
                                logging.warning(f"[S1 validation] DOYs in original files but missing from stack: {sorted(missing_doys)}")
                        if extra_doys:
                                logging.warning(f"[S1 validation] DOYs in stack but not in original files: {sorted(extra_doys)}")

                out_dir = Path(S2_DATA_PROCESSED) / tile_id
                out_dir.mkdir(parents=True, exist_ok=True)

                np.save(out_dir/"sar_ascending.npy", sar_asc)
                np.save(out_dir/"sar_ascending_doy.npy", asc_doys)
                np.save(out_dir/"sar_descending.npy", sar_desc)
                np.save(out_dir/"sar_descending_doy.npy", desc_doys)
                logging.info(f"[s1_stack] wrote final npy to {out_dir}")

                # remove raw S1
                remove_raw_s1(tile_id, S1_DATA_RAW)
                remove_raw_s1(tile_id, S1_DATA_PROCESSED)
        except Exception as e:
                logging.error(f"S1 stack failed: {e}")

if __name__ == "__main__":
        main()
