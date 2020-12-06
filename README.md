# dynamictiler_sample

```sh
pipenv install fastapi uvicorn aiofiles rio-tiler==2.0.0rc3
```

# dsm
## download dsm geotiff
[https://www.eorc.jaxa.jp/ALOS/aw3d30/index_j.htm](https://www.eorc.jaxa.jp/ALOS/aw3d30/index_j.htm)

save at cog directory.

## convert to cog
```
gdal_translate N035E138/ALPSMLC30_N035E138_DSM.tif temp.tif -co TILED=YES
gdaladdo -r nearest temp.tif 2 4 8 16
gdal_translate temp.tif ALPSMLC30_N035E138_DSM2.tif -co TILED=YES -co COPY_SRC_OVERVIEWS=YES
```


## exec
```
uvicorn app:app --reload
```

access to `http://127.0.0.1:8000/pages/dsm.html?max=1500&min=0&depth=15`