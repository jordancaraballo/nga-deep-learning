import xarray as xr  # read rasters
import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# -------------------------------------------------------------------------------
# module indices
# This class calculates remote sensing indices given xarray or numpy objects.
# Note: Most of our imagery uses the following set of bands.
# 8 band: ['CoastalBlue', 'Blue', 'Green', 'Yellow',
#          'Red', 'RedEdge', 'NIR1', 'NIR2']
# 4 band: ['Red', 'Green', 'Blue', 'NIR1', 'HOM1', 'HOM2']
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# Module Methods
# -------------------------------------------------------------------------------


def addindices(rastarr, bands, indices, factor=1.0) -> np.array:
    """
    :param rastarr: xarray or numpy array object in the form (c, h, w)
    :param bands: list with strings of bands in the raster
    :param indices: indices to calculate and append to the raster
    :param factor: factor used for toa imagery
    :return: raster with updated bands list
    """
    nbands = len(bands)  # get initial number of bands
    for indices_function in indices:  # iterate over each new band
        nbands += 1  # counter for number of bands

        # calculate band (indices)
        band, bandid = indices_function(rastarr, bands=bands, factor=factor)
        bands.append(bandid)  # append new band id to list of bands
        band.coords['band'] = [nbands]  # add band indices to raster
        rastarr = xr.concat([rastarr, band], dim='band')  # concat new band

    # update raster metadata, xarray attributes
    rastarr.attrs['scales'] = [rastarr.attrs['scales'][0]] * nbands
    rastarr.attrs['offsets'] = [rastarr.attrs['offsets'][0]] * nbands
    return rastarr, bands


# Difference Vegetation Index (DVI), type int16
def dvi(data, bands, factor=1.0, vtype='int16') -> np.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: list with strings of bands in the raster
    :param factor: factor used for toa imagery
    :return: new band with DVI calculated
    """
    # 8 and 4 band imagery: DVI := NIR1 - Red
    NIR1, Red = bands.index('NIR1'), bands.index('Red')
    return ((data[NIR1, :, :] / factor) - (data[Red, :, :] / factor)
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "DVI"


# Normalized Difference Vegetation Index (NDVI)
# range from +1.0 to -1.0, type float64
def ndvi(data, bands, factor=1.0, vtype='float64') -> np.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with NDVI calculated
    """
    # 8 and 4 band imagery: NDVI := (NIR - Red) / (NIR + RED)
    NIR1, Red = bands.index('NIR1'), bands.index('Red')
    return (((data[NIR1, :, :] / factor) - (data[Red, :, :] / factor)) /
            ((data[NIR1, :, :] / factor) + (data[Red, :, :] / factor))
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "NDVI"


# Forest Discrimination Index (FDI), type int16
def fdi(data, bands, factor=1.0, vtype='int16', device='CPU') -> np.array:
    """
    GPU Support
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with FDI calculated
    """
    # 8 band imagery: FDI := NIR2 - (RedEdge + Blue)
    # 4 band imagery: FDI := NIR1 - (Red + Blue)
    NIR = bands.index('NIR2') if 'NIR2' in bands else bands.index('NIR1')
    RED = bands.index('RedEdge') if 'RedEdge' in bands else bands.index('Red')
    BLUE = bands.index('Blue')

    fdi = data[NIR, :, :] - (data[RED, :, :] + data[BLUE, :, :])
    if device == 'CPU':
        fdi = fdi.expand_dims(dim="band", axis=0).fillna(0).astype(vtype)
    elif device == 'GPU':
        fdi = cp.nan_to_num(cp.expand_dims(fdi, 0))
    else:
        raise RuntimeError("{} device not supported".format(device))
    return fdi, "FDI"


# Shadow Index (SI), type int16
def si(data, bands, factor=1.0, vtype='int16', device='CPU') -> np.array:
    """
    GPU Support
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery:
    # SI := ((factor - Blue) * (factor - Green) * (factor - Red)) ** (1.0 / 3)
    BLUE, GREEN = bands.index('Blue'), bands.index('Green')
    RED = bands.index('Red')

    si = ((factor - data[BLUE, :, :]) * (factor - data[GREEN, :, :]) *
          (factor - data[RED, :, :])) ** (1.0/3.0)

    if device == 'CPU':
        si = si.expand_dims(dim="band", axis=0).fillna(0).astype(vtype)
    elif device == 'GPU':
        si = cp.nan_to_num(cp.expand_dims(si, 0))
    else:
        raise RuntimeError("{} device not supported".format(device))
    return si, "SI"


# Normalized Difference Water Index (DWI), type int16
def dwi(data, bands, factor=1.0, vtype='int16') -> np.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with DWI calculated
    """
    # 8 and 4 band imagery: DWI := factor * (Green - NIR1)
    Green, NIR1 = bands.index('Green'), bands.index('NIR1')
    return (factor * (data[Green, :, :] - data[NIR1, :, :])
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "DWI"


# Normalized Difference Water Index (NDWI), type int16
def ndwi(data, bands, factor=1.0, vtype='int16', device='CPU') -> np.array:
    """
    GPU Support
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery: NDWI := factor * (Green - NIR1) / (Green + NIR1)
    GREEN, NIR1 = bands.index('Green'), bands.index('NIR1')

    ndwi = factor * ((data[GREEN, :, :] - data[NIR1, :, :])
                     / (data[GREEN, :, :] + data[NIR1, :, :]))

    if device == 'CPU':
        ndwi = ndwi.expand_dims(dim="band", axis=0).fillna(0).astype(vtype)
    elif device == 'GPU':
        ndwi = cp.nan_to_num(cp.expand_dims(ndwi, 0))
    else:
        raise RuntimeError("{} device not supported".format(device))
    return ndwi, "NDWI"


# Shadow Index (SI), type float64
def cs1(data, bands, factor=1.0, vtype='float64') -> np.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with SI calculated
    """
    # 8 and 4 band imagery: CS1 := (3. * NIR1) / (Blue + Green + Red)
    NIR1, Blue = bands.index('NIR1'), bands.index('Blue')
    Green, Red = bands.index('Green'), bands.index('Red')
    return ((3.0 * (data[NIR1, :, :]/factor)) / (data[Blue, :, :]
            + data[Green, :, :] + data[Red, :, :])
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "CS1"


# Shadow Index (SI)
def cs2(data, bands, factor=1.0, vtype='int16') -> np.array:
    """
    :param data: xarray or numpy array object in the form (c, h, w)
    :param bands: number of the original bands of the raster
    :param factor: factor used for toa imagery
    :return: new band with CS2 calculated
    """
    # 8 and 4 band imagery: CS2 := (Blue + Green + Red + NIR1) / 4.
    NIR1, Blue = bands.index('NIR1'), bands.index('Blue')
    Green, Red = bands.index('Green'), bands.index('Red')
    return ((data[Blue, :, :] + data[Green, :, :] + data[Red, :, :]
            + data[NIR1, :, :]) / 4.0
            ).expand_dims(dim="band", axis=0).fillna(0).astype(vtype), "CS2"
