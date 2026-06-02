%%bash
export BUCKET="your-gcs-bucket-name"
export INPUT_PREFIX="input_prefix/"
export POLYGON_GEOJSON_BLOB="path/to/your/polygons.geojson"
export DATA_ROOT="data-root"
export METADATA_SUBREGIONS="path/to/your/subregions.geojson"
export WORK_DIR="/content/tile_work"
export MAX_WORKERS="2"
# export TARGET_TILES="10"
python /content/RTSmapping_v2/data/data_generation/negative_tile_creation.py