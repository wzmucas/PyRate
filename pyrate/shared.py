'''
Contains objects common to multiple parts of PyRate

Created on 12/09/2012
@author: Ben Davies
'''

import os
from numpy import where, nan, isnan, sum as nsum

try:
	from osgeo import gdal
	from gdalconst import GA_Update
except ImportError:
	import gdal

gdal.UseExceptions()

import roipac
from geodesy import cell_size


# Constants
AMPLITUDE_BAND = 1
PHASE_BAND = 2



class RasterBase(object):
	'''Base class for ROIPAC format raster datasets.'''

	def __init__(self, path, hdr_path=None):
		'''Handles common task of bundling various forms of ROIPAC header files with
		a binary data layer.'''

		if hdr_path:
			# handle non default header (eg. for look files with different naming)
			self.data_path, self.hdr_path = path, hdr_path
		else:
			# default the header path
			self.data_path, self.hdr_path = roipac.filename_pair(path)

		# dynamically include header items as instance attrs
		header = roipac.parse_header(self.hdr_path)
		self.__dict__.update(header)

		self.ehdr_path = None # path to EHdr format header
		self.dataset = None # for GDAL dataset obj
		self._readonly = not os.access(path, os.R_OK | os.W_OK)
		self.num_cells = self.FILE_LENGTH * self.WIDTH


	def __str__(self):
		name = self.__class__.__name__
		return "%s('%s')" % (name, self.data_path)


	def __repr__(self):
		name = self.__class__.__name__
		return "%s('%s', '%s')" % (name, self.data_path, self.hdr_path)


	def open(self, readonly=None):
		'''Opens generic raster dataset. Creates ESRI/EHdr format header in the data
		dir, creating a recogniseable header file for GDAL (as per ROIPAC doco).'''

		# default: open files as writeable, except if read only permissions are set
		if readonly not in [True, False, None]:
			raise ValueError("readonly must be True, False or None")

		if self._readonly is None:
			raise NotImplementedError

		if self._readonly is True:
			if readonly is False:
				raise IOError("Cannot open write protected file for writing")
			elif readonly is None:
				readonly = True # default to readonly as permissions are R/O

		if self.ehdr_path is None:
			self.ehdr_path = roipac.to_ehdr_header(self.hdr_path)
			args = (self.data_path,) if readonly else (self.data_path, GA_Update)
			self.dataset = gdal.Open(*args)

			if self.dataset is None:
				raise RasterException("Error opening %s" % self.data_path)

		else:
			if self.dataset is not None:
				msg = "open() already called for %s" % self
				raise RasterException(msg)

		if self.num_cells is None:
			self.num_cells = self.dataset.RasterYSize * self.dataset.RasterXSize

		#else:
		#	if self.num_cells != self.dataset.RasterYSize * self.dataset.RasterXSize:
		#		raise RasterException("GDAL Dataset size doesn't match header sizes")


	@property
	def shape(self):
		'''Returns tuple of (Y,X) shape of the raster (as per numpy.shape)'''
		return self.FILE_LENGTH, self.WIDTH


	@property
	def is_open(self):
		'''Returns True if the underlying dataset has been opened by GDAL'''
		return self.dataset is not None


	@property
	def is_read_only(self):
		return self._readonly


	def _get_band(self, band):
		'''
		Wrapper (with error checking) for GDAL's Band.GetRasterBand() method.
		band: number of band, starting at 1
		'''
		if self.dataset is not None:
			return self.dataset.GetRasterBand(band)
		else:
			raise RasterException("Raster %s has not been opened" % self.data_path)



class Ifg(RasterBase):
	"""Interferogram class, representing the difference between two acquisitions.
	Ifg objects double as a container for related data."""

	def __init__(self, path, hdr_path=None):
		'''Interferogram constructor, for 2 band ROIPAC Ifg raster datasets.'''
		RasterBase.__init__(self, path, hdr_path)
		self._amp_band = None
		self._phase_band = None
		self._phase_data = None

		self.X_CENTRE = self.WIDTH / 2
		self.Y_CENTRE = self.FILE_LENGTH / 2
		self.LAT_CENTRE = self.Y_FIRST + (self.Y_STEP * self.Y_CENTRE)
		self.LONG_CENTRE = self.X_FIRST + (self.X_STEP * self.X_CENTRE)

		# use cell size from centre of scene
		self.X_SIZE, self.Y_SIZE = cell_size(self.LAT_CENTRE, self.LONG_CENTRE,
		                                     self.X_STEP, self.Y_STEP)

		# creating code needs to set this flag after 0 -> NaN replacement
		self.nan_converted = False

		# TODO: what are these for?
		#self.max_variance = None # will be single floating point number
		#self.alpha = None # will be single floating point number


	def convert_to_nans(self, val=0):
		'''
		Converts given values in phase data to NaNs
		val - value to convert, default is 0
		'''
		self.phase_data = where(self.phase_data == val, nan, self.phase_data)
		self.nan_converted = True


	@property
	def amp_band(self):
		'''Returns a GDAL Band object for the amplitude band'''
		if self._amp_band is None:
			self._amp_band = self._get_band(AMPLITUDE_BAND)
		return self._amp_band


	@property
	def phase_band(self):
		'''Returns a GDAL Band object for the phase band'''
		if self._phase_band is None:
			self._phase_band = self._get_band(PHASE_BAND)
		return self._phase_band


	@property
	def phase_data(self):
		'''Returns entire phase band as an array'''
		if self._phase_data is None:
			self._phase_data = self.phase_band.ReadAsArray()
		return self._phase_data


	@phase_data.setter
	def phase_data(self, data):
		self._phase_data = data


	@property
	def phase_rows(self):
		'''Generator returning each row of the phase data'''
		for y in xrange(self.FILE_LENGTH):
			r = self.phase_band.ReadAsArray(yoff=y, win_xsize=self.WIDTH, win_ysize=1)
			yield r[0] # squeezes row from (1, WIDTH) to 1D array


	@property
	def nan_count(self):
		'''Returns number of NaN cells in the phase data'''
		return nsum(isnan(self.phase_data))


	@property
	def nan_fraction(self):
		'''Returns 0-1 (float) proportion of NaN cells for the phase band'''

		# don't cache nan_count as client code may modify phase data
		nan_count = self.nan_count

		# handle datasets with no 0 -> NaN replacement
		if self.nan_converted is False and nan_count == 0:
			nan_count = nsum(self.phase_data == 0)

		return nan_count / float(self.num_cells)


	def write_phase(self):
		'''Writes phase data to disk.'''

		if self.is_read_only:
			raise IOError("Cannot write to read only Ifg")

		self._phase_band.WriteArray(self.phase_data)


class Incidence(RasterBase):

	def __init__(self, path, hdr_path=None):
		'''Incidence obj constructor.'''
		RasterBase.__init__(self, path, hdr_path)
		self._incidence_band = None
		self._azimuth_band = None
		self._incidence_data = None
		self._azimuth_data = None


	@property
	def incidence_band(self):
		'''Returns the GDALBand for the incidence angle layer'''
		if self._incidence_band is None:
			self._incidence_band = self._get_band(1)
		return self._incidence_band


	@property
	def incidence_data(self):
		'''Returns the entire incidence band as an array'''
		if self._incidence_data is None:
			self._incidence_data = self.incidence_band.ReadAsArray()
		return self._incidence_data


	@property
	def azimuth_band(self):
		'''Returns the GDALBand for the azimuth layer'''
		if self._azimuth_band is None:
			self._azimuth_band = self._get_band(2)
		return self._azimuth_band


	@property
	def azimuth_data(self):
		'''Returns the entire incidence band as an array'''
		if self._azimuth_data is None:
			self._azimuth_data = self.azimuth_band.ReadAsArray()
		return self._azimuth_data



class DEM(RasterBase):
	"""Generic raster class for ROIPAC single band DEM files"""

	def __init__(self, path, hdr_path=None):
		'''DEM constructor.'''
		RasterBase.__init__(self, path, hdr_path)
		self._band = None


	@property
	def height_band(self):
		'''Returns the GDALBand for the elevation layer'''
		if self._band is None:
			self._band = self._get_band(1)
		return self._band



class IfgException(Exception):
	'''Generic exception class for interferogram errors'''
	pass

class RasterException(Exception):
	'''Generic exception for raster errors'''
	pass

class PyRateException(Exception):
	'''Generic exception class for PyRate S/W errors'''
	pass


class EpochList(object):
	'''Metadata container for epoch related information.'''

	def __init__(self, dates=None, repeat=None, spans=None):
		self.dates = dates # list of unique dates from all the ifgs
		self.repeat = repeat
		self.spans = spans # time span from earliest ifg


	def __str__(self):
		return "EpochList: %s" % str(self.dates)

	def __repr__(self):
		return "EpochList: %s" % repr(self.dates)