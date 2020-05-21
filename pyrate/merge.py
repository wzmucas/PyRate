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
stack rate and time series outputs and save as geotiff files
"""
from os.path import join, isfile
import pickle
import numpy as np
from osgeo import gdal
import subprocess
from pathlib import Path

from pyrate.core import shared, stack, ifgconstants as ifc, mpiops, config as cf
from pyrate.constants import REF_COLOR_MAP_PATH
from pyrate.core.logger import pyratelogger as log

gdal.SetCacheMax(64)


def main(params):
    """
    PyRate merge main function. Assembles product tiles in to
    single geotiff files
    """
    # setup paths
    rows, cols = params["rows"], params["cols"]
    mpiops.run_once(_merge_stack, rows, cols, params)
    mpiops.run_once(create_png_from_tif, params[cf.OUT_DIR])

    if params[cf.TIME_SERIES_CAL]:
        _merge_timeseries(rows, cols, params)
        #mpiops.run_once(_delete_tsincr_files, params)


def _merge_stack(rows, cols, params):
    """
    Merge stacking outputs
    """
    shape, tiles, ifgs_dict = _merge_setup(rows, cols, params)

    # read and assemble tile outputs
    rate = assemble_tiles(shape, params[cf.TMPDIR], tiles, out_type='stack_rate')
    error = assemble_tiles(shape, params[cf.TMPDIR], tiles, out_type='stack_error')
    samples = assemble_tiles(shape, params[cf.TMPDIR], tiles, out_type='stack_samples')

    # mask pixels according to threshold
    if params[cf.LR_MAXSIG] > 0:
        rate, error = stack.mask_rate(rate, error, params[cf.LR_MAXSIG])
    else:
        log.info('Skipping stack product masking (maxsig = 0)')

    # save geotiff and numpy array files
    _save_merged_files(ifgs_dict, params[cf.OUT_DIR], rate, out_type='stack_rate')
    _save_merged_files(ifgs_dict, params[cf.OUT_DIR], error, out_type='stack_error')
    _save_merged_files(ifgs_dict, params[cf.OUT_DIR], samples, out_type='stack_samples')


def _merge_timeseries(rows, cols, params):
    """
    Merge time series output
    """
    shape, tiles, ifgs_dict = _merge_setup(rows, cols, params)

    # load the first tsincr file to determine the number of time series tifs
    tsincr_file = join(params[cf.TMPDIR], 'tsincr_0.npy')
    tsincr = np.load(file=tsincr_file)
    # pylint: disable=no-member
    no_ts_tifs = tsincr.shape[2]

    # create 2 x no_ts_tifs as we are splitting tsincr and tscuml to all processes.
    process_tifs = mpiops.array_split(range(2 * no_ts_tifs))

    # depending on nvelpar, this will not fit in memory
    # e.g. nvelpar=100, nrows=10000, ncols=10000, 32bit floats need 40GB memory
    # 32 * 100 * 10000 * 10000 / 8 bytes = 4e10 bytes = 40 GB
    # the double for loop helps us overcome the memory limit
    log.info('Process {} writing {} timeseries tifs of '
             'total {}'.format(mpiops.rank, len(process_tifs), no_ts_tifs * 2))
    for i in process_tifs:
        if i < no_ts_tifs:
            tscum_g = assemble_tiles(shape, params[cf.TMPDIR], tiles, out_type='tscuml', index=i)
            _save_merged_files(ifgs_dict, params[cf.OUT_DIR], tscum_g, out_type='tscuml', index=i)
        else:
            i %= no_ts_tifs
            tsincr_g = assemble_tiles(shape, params[cf.TMPDIR], tiles, out_type='tsincr', index=i)
            _save_merged_files(ifgs_dict, params[cf.OUT_DIR], tsincr_g, out_type='tsincr', index=i)
    mpiops.comm.barrier()
    log.debug('Process {} finished writing {} timeseries tifs of '
             'total {}'.format(mpiops.rank, len(process_tifs), no_ts_tifs * 2))


def create_png_from_tif(output_folder_path):
    """
    Function to create a preview PNG format image from a geotiff
    """
    log.info('Creating quicklook image.')
    # open raster and choose band to find min, max
    raster_path = join(output_folder_path, "stack_rate.tif")

    if not isfile(raster_path):
        raise Exception("stack_rate.tif file not found at: "+raster_path)
    gtif = gdal.Open(raster_path)
    srcband = gtif.GetRasterBand(1)

    west, north, east, south = "", "", "", ""
    for line in gdal.Info(gtif).split('\n'):
        if "Upper Left" in line:
            west, north = line.split(")")[0].split("(")[1].split(",")
        if "Lower Right" in line:
            east, south = line.split(")")[0].split("(")[1].split(",")

    kml_file_path = join(output_folder_path, "stack_rate.kml")
    kml_file_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.1">
  <Document>
    <name>stack_rate.kml</name>
    <GroundOverlay>
      <name>stack_rate.png</name>
      <Icon>
        <href>stack_rate.png</href>
      </Icon>
      <LatLonBox>
        <north> """+north+""" </north>
        <south> """+south+""" </south>
        <east>  """+east+""" </east>
        <west>  """+west+""" </west>
      </LatLonBox>
    </GroundOverlay>
  </Document>
</kml>"""

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
        color_map_list[i+1][0] = str(no)

    color_map_path = join(output_folder_path, "colourmap.txt")
    with open(color_map_path, "w") as f:
        for i in range(no_of_data_value):
            f.write(' '.join(color_map_list[i]) + "\n")

    input_tif_path = join(output_folder_path, "stack_rate.tif")
    output_png_path = join(output_folder_path, "stack_rate.png")
    subprocess.check_call(["gdaldem", "color-relief", "-of", "PNG", input_tif_path, "-alpha",
                           color_map_path, output_png_path, "-nearest_color_entry"])
    log.debug('Finished creating quicklook image.')


def assemble_tiles(s, dir, tiles, out_type, index=None):
    """
    Function to reassemble tiles from numpy files in to a merged array

    :param tuple s: shape for merged array.
    :param str dir: path to directory containing numpy tile files.
    :param str out_type: product type string, used to construct numpy tile file name.
    :param int index: array third dimension index to extract from 3D time series array tiles.

    :return: merged_array: array assembled from all tiles.
    :rtype: ndarray
    """
    log.info('Re-assembling tiles for {}'.format(out_type))
    # pre-allocate dest array
    merged_array = np.empty(shape=s, dtype=np.float32)

    # loop over each tile, load and slot in to correct spot
    for t in tiles:
        tile_file = Path(join(dir, out_type + '_'+str(t.index)+'.npy'))
        tile = np.load(file=tile_file)
        if index is None: #2D array
            merged_array[t.top_left_y:t.bottom_right_y, t.top_left_x:t.bottom_right_x] = tile
        else: #3D array
            merged_array[t.top_left_y:t.bottom_right_y, t.top_left_x:t.bottom_right_x] = tile[:, :, index]

    log.debug('Finished assembling tiles for {}'.format(out_type))
    return merged_array


def _save_merged_files(ifgs_dict, outdir, array, out_type, index=None, savenpy=None):
    """
    Convenience function to save PyRate geotiff and numpy array files
    """
    log.info('Saving PyRate outputs {}'.format(out_type))
    gt, md, wkt = ifgs_dict['gt'], ifgs_dict['md'], ifgs_dict['wkt']
    epochlist = ifgs_dict['epochlist']

    if out_type in ('tsincr', 'tscuml'):
        epoch = epochlist.dates[index + 1]
        dest = join(outdir, out_type + "_" + str(epoch) + ".tif")
        # sequence position; first time slice is #0
        md['SEQUENCE_POSITION'] = index+1
        md[ifc.EPOCH_DATE] = epoch
    else:
        dest = join(outdir, out_type + ".tif")
        md[ifc.EPOCH_DATE] = epochlist.dates

    if out_type == 'stack_rate':
        md[ifc.DATA_TYPE] = ifc.STACKRATE
    elif out_type == 'stack_error':
        md[ifc.DATA_TYPE] = ifc.STACKERROR
    elif out_type == 'stack_samples':
        md[ifc.DATA_TYPE] = ifc.STACKSAMP
    elif out_type == 'tsincr':
        md[ifc.DATA_TYPE] = ifc.INCR
    else: #tscuml
        md[ifc.DATA_TYPE] = ifc.CUML

    shared.write_output_geotiff(md, gt, wkt, array, dest, np.nan)
    if savenpy:
        npy_rate_file = join(outdir, out_type + '.npy')
        np.save(file=npy_rate_file, arr=array)

    log.debug('Finished saving {}'.format(out_type))


def _merge_setup(rows, cols, params):
    """
    Convenience function for Merge set up steps
    """
    # setup paths
    xlks, _, crop = cf.transform_params(params)
    base_unw_paths = []

    for p in Path(params[cf.OUT_DIR]).rglob("*rlks_*cr.tif"):
        if "dem" not in str(p):
            base_unw_paths.append(str(p))

    if "tif" in base_unw_paths[0].split(".")[1]:
        dest_tifs = base_unw_paths # cf.get_dest_paths(base_unw_paths, crop, params, xlks)
        for i, dest_tif in enumerate(dest_tifs):
            dest_tifs[i] = dest_tif.replace("_tif", "")
    else:
        dest_tifs = base_unw_paths # cf.get_dest_paths(base_unw_paths, crop, params, xlks)

    # load previously saved preread_ifgs dict
    preread_ifgs_file = join(params[cf.TMPDIR], 'preread_ifgs.pk')
    ifgs_dict = pickle.load(open(preread_ifgs_file, 'rb'))
    ifgs = [v for v in ifgs_dict.values() if isinstance(v, shared.PrereadIfg)]
    shape = ifgs[0].shape
    tiles = shared.get_tiles(dest_tifs[0], rows, cols)
    return shape, tiles, ifgs_dict


def _delete_tsincr_files(params):
    """
    Convenience function to delete tsincr files
    """
    out_dir = Path(params[cf.OUT_DIR])
    for file_path in out_dir.iterdir():
        if "tsincr" in str(file_path):
            file_path.unlink()
