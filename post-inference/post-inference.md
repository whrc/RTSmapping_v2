Vectorization:

Feedback: rasterio.features.shapes or gdal_polygonize produces "stair-step" polygons. Apply a simplification algorithm (e.g., Douglas-Peucker) during vectorization to reduce file size and make the map look cleaner.