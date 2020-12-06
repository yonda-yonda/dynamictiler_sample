import os
import re
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

import uvicorn
from fastapi import FastAPI, Path, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from rasterio.crs import CRS
from starlette.background import BackgroundTask
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import Response

from rio_tiler.profiles import img_profiles
from rio_tiler.utils import render
from rio_tiler.io import COGReader
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.tasks import multi_arrays
from rio_tiler.models import ImageData
from rio_tiler.errors import TileOutsideBounds
from rio_tiler.colormap import cmap

import numpy as np

drivers = dict(jpg="JPEG", png="PNG")
mimetype = dict(png="image/png", jpg="image/jpg",)

class ImageType(str, Enum):
    png = "png"
    jpg = "jpg"

class TileResponse(Response):
    def __init__(
        self,
        content: bytes,
        media_type: str,
        status_code: int = 200,
        headers: dict = {},
        background: BackgroundTask = None,
        ttl: int = 3600,
    ) -> None:
        """Init tiler response."""
        headers.update({"Content-Type": media_type})
        if ttl:
            headers.update({"Cache-Control": "max-age=3600"})
        self.body = self.render(content)
        self.status_code = 200
        self.media_type = media_type
        self.background = background
        self.init_headers(headers)


app = FastAPI(
    title="rio-tiler",
    description="Cloud Optimized GeoTIFF tile server",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=0)

app.mount("/pages", StaticFiles(directory="pages"), name="static")

responses = {
    200: {
        "content": {"image/png": {}, "image/jpg": {}},
        "description": "Return an image.",
    }
}
tile_routes_params: Dict[str, Any] = dict(
    responses=responses, tags=["tiles"], response_class=TileResponse
)

dsm_assets = [
    "cog/ALPSMLC30_N035E138_DSM.tif",
    "cog/ALPSMLC30_N035E139_DSM.tif",
    "cog/ALPSMLC30_N035E140_DSM.tif",
    "cog/ALPSMLC30_N036E138_DSM.tif",
    "cog/ALPSMLC30_N036E139_DSM.tif",
    "cog/ALPSMLC30_N036E140_DSM.tif"]

def tiler(src_path: str, *args, **kwargs):
    with COGReader(src_path) as cog:
        return cog.tile(*args, **kwargs)

@app.get("/dsm/{z}/{x}/{y}", **tile_routes_params)
def dsm(
    z: int,
    x: int,
    y: int,
    min: Union[int, float] = Query(0, description="Minimum Height."),
    max: Union[int, float] = Query(8848, description="Maximum Height."),
):
    if -9999 >= min:
        min = 0
    if min >= max:
        max = 8848
    
    (tile, mask), _ = mosaic_reader(dsm_assets, tiler, x, y, z)
    if tile is None:
        raise HTTPException(status_code=404, detail="Tile Outside Bounds.")

    mask[tile[0] == -9999] = 0

    tile[tile < min] = min
    tile[tile > max] = max
    tile = (tile/(max - min) * 255).astype(np.uint8)

    format = ImageType.png
    driver = drivers[format.value]
    options = img_profiles.get(driver.lower(), {})

    colormap = cmap.get('jet')
    img = render(tile, mask, img_format=driver, **options, colormap=colormap)

    return TileResponse(img, media_type=mimetype[format.value])

water =  np.array([
    76 * np.ones((256, 256), dtype=np.uint8), 
    108 * np.ones((256, 256), dtype=np.uint8), 
    179 * np.ones((256, 256), dtype=np.uint8)])

@app.get("/flood/{z}/{x}/{y}", **tile_routes_params)
def flood(
    z: int,
    x: int,
    y: int,
    depth: Union[int, float] = Query(0, description="Water Depth."),
):
    if -9999 >= depth:
        depth = 0
 
    (tile, mask), _ = mosaic_reader(dsm_assets, tiler, x, y, z)
    if tile is None:
        raise HTTPException(status_code=404, detail="Tile Outside Bounds.")
    mask[tile[0] == -9999] = 0
    mask[tile[0] > depth] = 0

    format = ImageType.png
    driver = drivers[format.value]
    options = img_profiles.get(driver.lower(), {})

    img = render(water, mask, img_format=driver, **options)

    return TileResponse(img, media_type=mimetype[format.value])


@app.get("/landsat8_tile/{z}/{x}/{y}", **tile_routes_params)
def landsat8_tile(
    z: int,
    x: int,
    y: int,
    base_url: str = Query(..., description="Landsat-8 Base URL."),
    a: Union[int, float] = Query(40, description="Coefficient 'a' Of Sigmoid Filter."),
    b: Union[int, float] = Query(0.15, description="Coefficient 'b' Of Sigmoid Filter."),
):
    mlt_path = '{}_MTL.txt'.format(base_url)
    if base_url.startswith('http') or base_url.startswith('//:'):
        r = requests.get(mlt_path)
        mlt = r.text
    else:
        with open(mlt_path) as f:
            mlt = f.read()

    REFLECTANCE_MULT_BANDS = re.findall('REFLECTANCE_MULT_BAND_\d+\s+=\s+(.*)\n', mlt)
    REFLECTANCE_ADD_BANDS = re.findall('REFLECTANCE_ADD_BAND_\d+\s+=\s+(.*)\n', mlt)
    SUN_ELEVATION = re.findall('SUN_ELEVATION\s+=\s+(.*)\n', mlt)
    denominator = math.sin(math.radians(float(SUN_ELEVATION[0])))

    bands = [{
            'url':'{}_B{}.TIF'.format(base_url, band),
            'mult': float(REFLECTANCE_MULT_BANDS[band - 1]),
            'add': float(REFLECTANCE_ADD_BANDS[band - 1]),
            'denominator': denominator
        } for band in [4,3,2]]

    def _reader(band, *args: Any, **kwargs: Any):
        with COGReader(band['url']) as cog:
            try:
                tile, mask = cog.tile(x, y, z, tilesize=256)
                mask[tile[0] == 0] = 0
                tile = (tile * band['mult'] + band['add'])/band['denominator']
                tile[tile < 0] = 0                
                return ImageData(tile, mask)
            except TileOutsideBounds as e:
                raise HTTPException(status_code=404, detail=str(e))

    output = multi_arrays(bands, _reader)

    f0 = 1/(1 + np.exp(a*b))
    f1 = 1/(1 + np.exp(a*(b - 1)))
    tile = ((255/(1 + np.exp(a*(b - output.data))) - f0) / (f1 - f0)).astype(np.uint8)
    mask = output.mask

    driver = "PNG"
    options = img_profiles.get(driver.lower(), {})
    img = render(tile, mask, img_format=driver, **options)

    return TileResponse(img, media_type="image/png")