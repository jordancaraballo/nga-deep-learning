import os
import glob
import cv2
from tqdm import tqdm
import xarray as xr
import numpy as np  # for arrays modifications
from skimage import color
import rasterio.features as riofeat  # rasterio features include sieve filter
import rasterio as rio
from scipy.ndimage import median_filter, binary_fill_holes

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"


def npy_to_tif(raster_f='image.tif', segments='segment.npy',
               outtif='segment.tif', ndval=-9999
               ):
    """
    Args:
        raster_f:
        segments:
        outtif:
    Returns:
    """
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    print(meta)

    # load numpy array if file is given
    if type(segments) == str:
        segments = np.load(segments)
    segments = segments.astype('int16')
    print(segments.dtype)  # check datatype

    nodatavals[nodatavals == 0] = ndval
    segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(outtif, 'w', **out_meta) as dst:
        dst.write(segments, 1)


def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    try:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    except ZeroDivisionError:
        cx, cy = 0, 0

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


def main():

    # CNN
    # root_dir = '/att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/vietnam_cm_models/predictions/100_unet_viet_cm_4chann_std_Adadelta_256_0.0001_128_1000.h5'
    # out_dir = f'{root_dir}/processed'
    
    # RF
    #root_dir = '/att/gpfsfs/atrepo01/ILAB/projects/Vietnam/Jordan/Vietnam_CLOUD/cloud_predictions_rf/cloudmask_keelin_4bands_rf'
    #out_dir = f'{root_dir}/processed_rf'

    root_dir = '/Users/jacaraba/Documents/Development/ILAB/ai-cloud-shadow-masking-vhr/projects/cnn_cloud_v3/data/alaska/predictions'
    out_dir = '/Users/jacaraba/Documents/Development/ILAB/ai-cloud-shadow-masking-vhr/projects/cnn_cloud_v3/data/alaska/predictions/post'

    os.system('mkdir -p {}'.format(out_dir))

    #mask_filenames = f'{root_dir}/images/*.tif'
    mask_filenames = f'{root_dir}/*.tif'
    mask_filenames = glob.glob(mask_filenames)

    for mask_f in tqdm(mask_filenames):

        # open mask
        raster_mask = xr.open_rasterio(mask_f, chunks={'band': 1, 'x': 2048, 'y': 2048})

        # out filename
        out_filename = out_dir + '/' + mask_f.split('/')[-1]

        output = np.squeeze(raster_mask.values)  # values

        # riofeat.sieve(prediction, size, out, mask, connectivity)
        riofeat.sieve(output, 800, output, None, 8)

        # median filter and binary filter
        output = median_filter(output, size=20)
        output = binary_fill_holes(output).astype(int)

        # increase size of contour
        output = np.uint8(np.squeeze(output) * 255)
        ret, thresh = cv2.threshold(output, 127, 255, 0)

        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        conts = np.zeros((thresh.shape[0], thresh.shape[1], 3))

        for c in contours:
            c = scale_contour(c, 1.12)  # 1.05 looks decent
            conts = cv2.drawContours(conts, [c], -1, (0, 255, 0), cv2.FILLED)

        # converting mask into raster
        conts = color.rgb2gray(conts).astype('int16')
        conts[conts > 0] = 1

        npy_to_tif(raster_f=mask_f, segments=conts, outtif=out_filename)


# -------------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    main()
