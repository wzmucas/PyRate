#   This Python module is part of the PyRate software package.
#
#   Copyright 2020 Geoscience Australia
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""
This Python module does post-processing steps to assemble the
rate and time series outputs and save as geotiff files
"""
import pickle as cp
import subprocess
from os.path import join

from constants import REF_COLOR_MAP_PATH
from core import shared, ifgconstants as ifc, mpiops, config as cf
from core.logger import pyratelogger as log
from core.shared import PrereadIfg
import os
from osgeo import gdal, gdalconst
from configuration import Configuration
import numpy as np
import math
import pathlib

gdal.SetCacheMax(64)

# Constants
MASTER_PROCESS = 0


def main(params):
    """PyRate merge main function. Assembles product tiles in to single geotiff
    files
    
    Args:
    
    Args:

    Args:
      params: 

    Returns:
      

    """
    # setup paths
    rows, cols = params["rows"], params["cols"]
    _merge_stack(rows, cols, params)
    if params[cf.TIME_SERIES_CAL]:
        _merge_timeseries(rows, cols, params)

    log.info("Creating quicklook images.")
    output_folder_path = os.path.dirname(params["tmpdir"])
    create_png_from_tif(output_folder_path)
    log.debug("Finished creating quicklook images.")

    log.debug("Start adding ref pixel information to geotiff metadata")
    sampled_interferogram = params["interferogram_files"][0].sampled_path
    source_dataset = gdal.Open(sampled_interferogram, gdal.GA_Update)
    source_metadata = source_dataset.GetMetadata()
    pyrate_refpix_x = int(source_metadata["PYRATE_REFPIX_X"])
    pyrate_refpix_y = int(source_metadata["PYRATE_REFPIX_Y"])
    pyrate_refpix_lat = float(source_metadata["PYRATE_REFPIX_LAT"])
    pyrate_refpix_lon = float(source_metadata["PYRATE_REFPIX_LON"])
    wavelength = float(source_metadata["WAVELENGTH_METRES"])
    # manual close dataset
    source_dataset = None
    del source_dataset

    tscuml_files = list(pathlib.Path(params["outdir"]).glob('tscuml_*.tif'))
    update_ifg_metadata(tscuml_files, pyrate_refpix_x, pyrate_refpix_y, pyrate_refpix_lat, pyrate_refpix_lon,
                        wavelength, params)
    stack_file = list(pathlib.Path(params["outdir"]).glob('stack_*.tif'))
    update_ifg_metadata(stack_file, pyrate_refpix_x, pyrate_refpix_y, pyrate_refpix_lat, pyrate_refpix_lon, wavelength,
                        params)


def update_ifg_metadata(dataset_paths, pyrate_refpix_x, pyrate_refpix_y, pyrate_refpix_lat, pyrate_refpix_lon, wavelength, params):
    for dataset_path in dataset_paths:
        dataset_path = str(dataset_path)
        log.debug("Updating metadata for: "+dataset_path)

        dataset = gdal.Open(dataset_path, gdalconst.GA_Update)
        phase_data = dataset.GetRasterBand(1).ReadAsArray()

        log.debug("Update no data values in dataset")

        # convert to nans
        phase_data = np.where(np.isclose(phase_data, params["noDataValue"], atol=1e-6), np.nan, phase_data)

        # convert radians to mm
        MM_PER_METRE = 1000
        phase_data = phase_data * MM_PER_METRE * (wavelength / (4 * math.pi))

        half_patch_size = params["refchipsize"] // 2
        x, y = pyrate_refpix_x, pyrate_refpix_y

        log.debug("Extract reference pixel windows")
        data = np.array(phase_data)[y - half_patch_size: y + half_patch_size + 1, x - half_patch_size: x + half_patch_size + 1]

        log.debug("Calculate standard deviation for reference window")
        standard_deviation_ref_area = np.std(data[~np.isnan(data)])
        mean_ref_area = np.mean(data[~np.isnan(data)])

        metadata = dataset.GetMetadata()
        metadata.update({
            'NAN_STATUS': "CONVERTED",
            'DATA_UNITS': "MILLIMETRES",
            'PYRATE_REFPIX_X': str(pyrate_refpix_x),
            'PYRATE_REFPIX_Y': str(pyrate_refpix_y),
            'PYRATE_REFPIX_LAT': str(pyrate_refpix_lat),
            'PYRATE_REFPIX_LON': str(pyrate_refpix_lon),
            'PYRATE_MEAN_REF_AREA': str(mean_ref_area),
            'PYRATE_STANDARD_DEVIATION_REF_AREA': str(standard_deviation_ref_area)
        })
        dataset.SetMetadata(metadata)

        # manual close dataset
        dataset = None
        del dataset

def create_png_from_tif(output_folder_path):
    """

    Args:
      output_folder_path: 

    Returns:

    """
    # open raster and choose band to find min, max
    """
    Args:
        output_folder_path:
    """
    raster_path = os.path.join(output_folder_path, "stack_rate.tif")

    if not os.path.isfile(raster_path):
        raise Exception("stack_rate.tif file not found at: " + raster_path)
    gtif = gdal.Open(raster_path)
    srcband = gtif.GetRasterBand(1)

    west, north, east, south = "", "", "", ""
    for line in gdal.Info(gtif).split("\n"):
        if "Upper Left" in line:
            west, north = line.split(")")[0].split("(")[1].split(",")
        if "Lower Right" in line:
            east, south = line.split(")")[0].split("(")[1].split(",")

    kml_file_path = os.path.join(output_folder_path, "stack_rate.kml")
    kml_file_content = (
            """<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://earth.google.com/kml/2.1">
      <Document>
        <name>stack_rate.kml</name>
        <GroundOverlay>
          <name>stack_rate.png</name>
          <Icon>
            <href>stack_rate.png</href>
          </Icon>
          <LatLonBox>
            <north> """
            + north
            + """ </north>
        <south> """
            + south
            + """ </south>
        <east>  """
            + east
            + """ </east>
        <west>  """
            + west
            + """ </west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>"""
    )

    with open(kml_file_path, "w") as f:
        f.write(kml_file_content)

    # Get raster statistics
    minimum, maximum, mean, stddev = srcband.GetStatistics(True, True)
    maximum = max(abs(minimum), abs(maximum))
    minimum = -1 * maximum
    step = (maximum - minimum) / 256.0

    del gtif  # manually close raster

    # read color map from utilities and write it to the output folder

    with open(REF_COLOR_MAP_PATH, "r") as f:
        color_map_list = []
        for line in f.readlines():
            color_map_list.append(line.strip().split(" "))

    no_of_data_value = len(np.arange(minimum, maximum, step))
    for i, no in enumerate(np.arange(minimum, maximum, step)):
        color_map_list[i + 1][0] = str(no)

    color_map_path = os.path.join(output_folder_path, "colormap.txt")
    with open(color_map_path, "w") as f:
        for i in range(no_of_data_value):
            f.write(" ".join(color_map_list[i]) + "\n")

    input_tif_path = os.path.join(output_folder_path, "stack_rate.tif")
    output_png_path = os.path.join(output_folder_path, "stack_rate.png")
    subprocess.check_call(
        ["gdaldem", "color-relief", "-of", "PNG", input_tif_path, "-alpha", color_map_path, output_png_path,
         "-nearest_color_entry"]
    )


def _merge_stack(rows, cols, params):
    """Merge stacking outputs
    
    Args:
        rows:
        cols:
    
    Args:
      rows: param cols:

    Args:
      cols: 
      rows: 
      params: 

    Returns:
      

    """
    # setup paths
    interferogram_file_paths = []
    for interferogram_file_path in params["interferogram_files"]:
        interferogram_file_paths.append(interferogram_file_path.sampled_path)

    # load previously saved prepread_ifgs dict
    preread_ifgs_file = join(params[cf.TMPDIR], "preread_ifgs.pk")
    ifgs = cp.load(open(preread_ifgs_file, "rb"))
    tiles = shared.get_tiles(interferogram_file_paths[0], rows, cols)

    # stacking aggregation
    if mpiops.size >= 3:
        [_save_stack(ifgs, params, tiles, out_type=t) for i, t in
         enumerate(["stack_rate", "stack_error", "stack_samples"]) if i == mpiops.rank]
    else:
        if mpiops.rank == MASTER_PROCESS:
            [_save_stack(ifgs, params, tiles, out_type=t) for t in ["stack_rate", "stack_error", "stack_samples"]]


def _save_stack(ifgs_dict, params, tiles, out_type):
    """Save stacking outputs
    
    Args:
        ifgs_dict:
    
    Args:
      tiles: param out_type:
      ifgs_dict: param params:

    Args:
      out_type: 
      ifgs_dict: 
      params: 
      tiles: 

    Returns:
      

    """
    log.info("Merging PyRate outputs {}".format(out_type))
    gt, md, wkt = ifgs_dict["gt"], ifgs_dict["md"], ifgs_dict["wkt"]
    epochlist = ifgs_dict["epochlist"]
    ifgs = [v for v in ifgs_dict.values() if isinstance(v, PrereadIfg)]
    dest = os.path.join(params[cf.OUT_DIR], out_type + ".tif")
    md[ifc.EPOCH_DATE] = epochlist.dates
    if out_type == "stack_rate":
        md[ifc.DATA_TYPE] = ifc.STACKRATE
    elif out_type == "stack_error":
        md[ifc.DATA_TYPE] = ifc.STACKERROR
    else:
        md[ifc.DATA_TYPE] = ifc.STACKSAMP

    rate = np.zeros(shape=ifgs[0].shape, dtype=np.float32)

    for t in tiles:
        rate_file = os.path.join(params[cf.TMPDIR], out_type + "_" + str(t.index) + ".npy")
        rate_file = pathlib.Path(rate_file)
        rate_tile = np.load(file=rate_file)
        rate[t.top_left_y: t.bottom_right_y, t.top_left_x: t.bottom_right_x] = rate_tile
    shared.write_output_geotiff(md, gt, wkt, rate, dest, np.nan)
    npy_rate_file = os.path.join(params[cf.OUT_DIR], out_type + ".npy")
    np.save(file=npy_rate_file, arr=rate)

    log.debug("Finished PyRate merging {}".format(out_type))


def _merge_timeseries(rows, cols,  params):
    """Merge time series output
    
    Args:
        rows:
        cols:
    
    Args:
      rows: param cols:

    Args:
      cols: 
      rows: 
      params: 

    Returns:
      

    """

    dest_tifs = []
    for interferogram in params["interferogram_files"]:
        dest_tifs.append(interferogram.sampled_path)

    output_dir = params[cf.TMPDIR]
    # load previously saved prepread_ifgs dict
    preread_ifgs_file = join(output_dir, "preread_ifgs.pk")
    ifgs = cp.load(open(preread_ifgs_file, "rb"))

    # metadata and projections
    gt, md, wkt = ifgs["gt"], ifgs["md"], ifgs["wkt"]
    epochlist = ifgs["epochlist"]
    ifgs = [v for v in ifgs.values() if isinstance(v, PrereadIfg)]

    tiles = shared.get_tiles(dest_tifs[0], rows, cols)

    # load the first tsincr file to determine the number of time series tifs
    tsincr_file = os.path.join(output_dir, "tsincr_0.npy")

    tsincr = np.load(file=tsincr_file)

    no_ts_tifs = tsincr.shape[2]
    # we create 2 x no_ts_tifs as we are splitting tsincr and tscuml
    # to all processes.
    process_tifs = mpiops.array_split(range(2 * no_ts_tifs))

    # depending on nvelpar, this will not fit in memory
    # e.g. nvelpar=100, nrows=10000, ncols=10000, 32bit floats need 40GB memory
    # 32 * 100 * 10000 * 10000 / 8 bytes = 4e10 bytes = 40 GB
    # the double for loop helps us overcome the memory limit
    log.info("Process {} writing {} ts (incr/cuml) tifs of " "total {}".format(mpiops.rank, len(process_tifs),
                                                                               no_ts_tifs * 2))
    for i in process_tifs:
        tscum_g = np.empty(shape=ifgs[0].shape, dtype=np.float32)
        if i < no_ts_tifs:
            for n, t in enumerate(tiles):
                _assemble_tiles(i, n, t, tscum_g, output_dir, "tscuml")
            md[ifc.EPOCH_DATE] = epochlist.dates[i + 1]
            # sequence position; first time slice is #0
            md["SEQUENCE_POSITION"] = i + 1
            dest = os.path.join(params[cf.OUT_DIR], "tscuml" + "_" + str(epochlist.dates[i + 1]) + ".tif")
            md[ifc.DATA_TYPE] = ifc.CUML
            shared.write_output_geotiff(md, gt, wkt, tscum_g, dest, np.nan)
        else:
            tsincr_g = np.empty(shape=ifgs[0].shape, dtype=np.float32)
            i %= no_ts_tifs
            for n, t in enumerate(tiles):
                _assemble_tiles(i, n, t, tsincr_g, output_dir, "tsincr")
            md[ifc.EPOCH_DATE] = epochlist.dates[i + 1]
            # sequence position; first time slice is #0
            md["SEQUENCE_POSITION"] = i + 1
            dest = os.path.join(params[cf.OUT_DIR], "tsincr" + "_" + str(epochlist.dates[i + 1]) + ".tif")
            md[ifc.DATA_TYPE] = ifc.INCR
            shared.write_output_geotiff(md, gt, wkt, tsincr_g, dest, np.nan)
    log.debug("Process {} finished writing {} ts (incr/cuml) tifs of " "total {}".format(
        mpiops.rank, len(process_tifs), no_ts_tifs * 2))


def _assemble_tiles(i, n, tile, tsincr_g, output_dir, outtype):
    """A reusable time series tile assembly function

    Args:
      i: param n:
      tile: param tsincr_g:
      output_dir: param outtype:
      n: 
      tsincr_g: 
      outtype: 

    Returns:

    """
    tsincr_file = os.path.join(output_dir, "{}_{}.npy".format(outtype, n))
    tsincr = np.load(file=tsincr_file)
    tsincr_g[tile.top_left_y: tile.bottom_right_y, tile.top_left_x: tile.bottom_right_x] = tsincr[:, :, i]
