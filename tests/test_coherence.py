import os
import unittest
import pytest
import glob
import copy

from . import common

import core.config as cf
import conv2tif, prepifg
from core import gdal_python

import os
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
from osgeo import gdalconst
from osgeo import gdal_array
import numpy as np
from osgeo import osr


class CoherenceMaskingTest(unittest.TestCase):
    """ """
    def test_coherence_files_not_converted(self):
        """ """
        # define constants
        NO_DATA_VALUE = 0
        driver = gdal.GetDriverByName("GTiff")

        # create a sample gdal dataset

        # sample gdal dataset
        sample_gdal_filename = "sample_gdal_dataset.tif"
        sample_gdal_dataset = driver.Create(sample_gdal_filename, 5, 5, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        wkt_projection = srs.ExportToWkt()
        sample_gdal_dataset.SetProjection(wkt_projection)

        sample_gdal_band = sample_gdal_dataset.GetRasterBand(1)
        sample_gdal_band.SetNoDataValue(NO_DATA_VALUE)
        sample_gdal_band.WriteArray(np.arange(25).reshape(5, 5))

        # create a coherence mask dataset
        coherence_mask_filename = "coherence_mask_dataset.tif"
        coherence_mask_dataset = driver.Create(coherence_mask_filename, 5, 5, 1, gdal.GDT_Float32)
        srs = osr.SpatialReference()
        wkt_projection = srs.ExportToWkt()
        coherence_mask_dataset.SetProjection(wkt_projection)
        coherence_mask_band = coherence_mask_dataset.GetRasterBand(1)
        coherence_mask_band.SetNoDataValue(NO_DATA_VALUE)
        coherence_mask_band.WriteArray(np.arange(0, 75, 3).reshape(5, 5) / 100.0)

        # create a artificial masked dataset
        expected_result_array = np.nan_to_num(
            np.array(
                [
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            )
        )

        # use the gdal_python.coherence_masking to find the actual mask dataset
        threshold = 0.3
        gdal_python.coherence_masking(sample_gdal_dataset, coherence_mask_dataset, threshold)
        sample_gdal_array = np.nan_to_num(sample_gdal_dataset.GetRasterBand(1).ReadAsArray())

        # compare the artificial masked and actual masked datasets
        self.assertTrue(np.array_equiv(sample_gdal_array, expected_result_array))

        # del the tmp datasets created
        del coherence_mask_dataset
        os.remove(coherence_mask_filename)

        del sample_gdal_dataset
        os.remove(sample_gdal_filename)


if __name__ == "__main__":
    unittest.main()
