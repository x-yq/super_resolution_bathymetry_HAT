"""
GeoTIFF I/O utilities for the project.

This component was jointly developed by:
    Niklas Schmolenski
"""

from osgeo import gdal
import numpy as np

def read_geotiff3D(filename, bathymetry=False):
    ds = gdal.Open(filename)

    # Dimensions
    # print("X-Size: ", ds.RasterXSize)
    # print("Y-Size: ", ds.RasterYSize)

    # Number of bands
    # print("Bands: ", ds.RasterCount)

    # Metadata for the raster dataset
    # print("Meta Data: ", ds.GetMetadata())

    
    band1 = ds.GetRasterBand(1)
    if not bathymetry:
        band2 = ds.GetRasterBand(2)
        band3 = ds.GetRasterBand(3)
        bands = [band1.ReadAsArray(), band2.ReadAsArray(), band3.ReadAsArray()]
    else:
        bands = [band1.ReadAsArray()]
    img = np.stack(bands, axis=0)
    img = np.transpose(img, (1, 2, 0)).astype(np.float32)
    
    return img, ds


def write_geotiff3D(filename, img, in_ds, bathymetry=False):
    img = np.transpose(img, (2, 0, 1)) # H x W x C -> C x H x W

    if img.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")

    if bathymetry:
        out_ds = driver.Create(filename, img.shape[1], img.shape[0], 1, arr_type)
    else:
        out_ds = driver.Create(filename, img[0].shape[1], img[0].shape[0], 3, arr_type)

    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())

    if not bathymetry:
        band1 = out_ds.GetRasterBand(1)
        band2 = out_ds.GetRasterBand(2)
        band3 = out_ds.GetRasterBand(3)
        band1.WriteArray(img[0])
        band1.FlushCache()
        band1.ComputeStatistics(False)
        band2.WriteArray(img[1])
        band2.FlushCache()
        band2.ComputeStatistics(False)
        band3.WriteArray(img[2])
        band3.FlushCache()
        band3.ComputeStatistics(False)
    else:
        band1 = out_ds.GetRasterBand(1)
        band1.WriteArray(img)
        band1.FlushCache()
        band1.ComputeStatistics(False)
