%%bash
export BUCKET="bucket"
export INPUT_PREFIX="tiles/input/"
export DATA_ROOT="tiles/"
export POSITIVE_GEOJSON="training_labels"
export IGNORE_GEOJSON="ignore_labels"
export METADATA_SUBREGIONS="metadata_subregions"
export WORK_DIR="/content/work"
export MAX_WORKERS=4
# export TEST_LIMIT=10
python /content/RTSmapping_v2/data/data_generation/positive_tile_creation.py