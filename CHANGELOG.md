## 0.2.0 (2025-05-12)

### Fix

- **workflow**: update dependency sync step to set up Python environment
- **pyproject**: correct package name from 'perdictor' to 'predictor'
- **workflow**: update dependency sync step and adjust Python installation command
- **workflow**: correct command to install project dependencies in unit test workflow
- **workflow**: add step to install project dependencies in unit test workflow
- **import**: fixes requests error
- **dockerfile**: fixes bug on dockerfile building after uv migration

### Refactor

- **tests**: migrate unit tests from unittest to pytest and clean up test structure
- **tests**: improve readability of predict function calls in test cases

## v0.3.15 (2026-01-20)

### Fix

- **validate**: fixes bug on null features validation

## v0.3.14 (2026-01-20)

### Fix

- **test**: fixes the test case

## v0.3.13 (2026-01-20)

### Fix

- **point**: fixes point predictions

## v0.3.12 (2026-01-18)

### Fix

- **predictions**: fixes the filepath on the prediction output

## v0.3.11 (2026-01-17)

### Fix

- **lock**: scikit image

## v0.3.10 (2025-12-24)

### Fix

- add morphological cleaning function and integrate it into prediction workflow
- update prediction output path and clean up morphological mask function

## v0.3.9 (2025-09-25)

### Fix

- **geom**: fixes geom validation issue

## v0.3.8 (2025-08-03)

### Fix

- **bing**: adds support for bign

## v0.3.7 (2025-08-03)

### Fix

- **python**: fixes geomlversion

## v0.3.6 (2025-07-06)

### Fix

- **predict**: update temporary directory handling for mask vectorization
- **fix-basepath**: basepath sync
- **fixes-metadata-cleanup**: fixes metadata cleanup problem also introduces an ability to keep all runtime metadata
- **predict**: update output path handling and directory structure for predictions

## v0.3.5 (2025-07-06)

### Fix

- **fixes-overcorrection-of-building-rotation**: fixes building rotation with new threshold

## v0.3.4 (2025-05-24)

### Fix

- **import**: fixes yoloseg import error

## v0.3.3 (2025-05-24)

### Fix

- **module**: fixes build module

## v0.3.2 (2025-05-22)

### Fix

- **build**: fixes package build

## v0.3.1 (2025-05-22)

### Fix

- **bump**: versions compability

## v0.3.0 (2025-05-12)

### Fix

- **version**: fixes toml version
- **workflow**: update dependency sync step to set up Python environment
- **pyproject**: correct package name from 'perdictor' to 'predictor'
- **workflow**: update dependency sync step and adjust Python installation command
- **workflow**: correct command to install project dependencies in unit test workflow
- **workflow**: add step to install project dependencies in unit test workflow
- **import**: fixes requests error
- **dockerfile**: fixes bug on dockerfile building after uv migration

### Refactor

- **tests**: migrate unit tests from unittest to pytest and clean up test structure
- **tests**: improve readability of predict function calls in test cases

## v0.1.13 (2025-05-07)

### Fix

- **geojson**: add support for the filepath

## v0.1.12 (2025-05-07)

### Fix

- **models**: introduces models to incorporate other developments on top

## v0.1.11 (2025-05-07)

### Fix

- **bump**: bumps recent version of predictor

## v0.1.10 (2025-03-29)

### Fix

- **cleanup**: cleanups the predictions

## v0.1.9 (2025-03-29)

### Fix

- **ops**: vectorize ops

## v0.1.8 (2025-03-27)

### Fix

- **docker**: fixes bug on the prod build for the local lib installation

## v0.1.7 (2025-03-27)

### Fix

- **consistency**: added backward compatible api changes

## v0.1.6 (2025-03-27)

### Fix

- **typo**: fixes typo in env variable main
- update README example to reflect correct import and usage for predictions

## v0.1.5 (2025-03-27)

### Fix

- **defaultmodelpath**: fixes bug with model path url

## v0.1.4 (2025-03-27)

### Fix

- **ci**: fixes env variabel in ci
- update README to use asyncio for prediction execution

## v0.1.3 (2025-03-27)

### Fix

- **ci**: fixes python version as per the poetry min python

## v0.1.2 (2025-03-27)

### Fix

- **ci**: regenerates lockfile

## v0.1.1 (2025-03-27)

### Fix

- **version**: build version fix

## v0.1.0 (2025-03-27)

### Feat

- **predictor**: add default model URLs for YOLO and RAMP in predictor module
- **api**: add root endpoint to return API information and documentation links
- **geomltoolkits**: integration with geomltoolkits for modularization

### Fix

- **default**: adds default value for model checkpoitns and image url
- **prediction**: update checkpoint URL in PredictionRequest and set default tms_url in predict function
- **prediction**: update default vectorization algorithm to use environment variable
- **prediction**: update source URL format and improve error handling in prediction API
- **prediction**: update source URL for tile retrieval in PredictionRequest
- **fix-tests**: fixes
- **predictor**: enable georeferencing and update CRS to 3857 in prediction functions
- **predictor**: rename output variable for clarity in TFLite prediction
- **tile**: fixes tile overlap issue

### Refactor

- **doc**: added documentation changes for default url
- **doc**: adds documentation and builds

## v0.0.39 (2024-11-26)

### Refactor

- **prediction**: remove bbox feature removal logic
- **vectorizer**: filter out background polygons during vectorization

## v0.0.38 (2024-11-26)

### Fix

- **bbox**: remove the bbox in predictions

## v0.0.37 (2024-11-25)

### Fix

- **build**: loosen version of geopandas

## v0.0.36 (2024-11-25)

### Feat

- **yoloseg**: add YOLOSeg integration and update prediction methods
- **docker**: restructure Docker setup and update requirements
- **prediction**: enhance predict_keras to process image batches and save masks for predictions
- **prediction**: update predict_tflite to handle image paths and save masks for predictions
- **prediction**: refactor prediction functions to handle batch processing and save masks
- **dependencies**: add ONNX and ONNX Runtime support; downgrade Python version to 3.10
- **predictor**: add ONNX model support and update prediction functions
- **prediction**: enhance model loading and prediction functions for YOLO, Keras, and TFLite

### Fix

- **prediction**: improve error handling for TensorFlow and TFLite imports
- **test-case**: disable h5 fileformat for now
- **workflow**: update Python version to 3.9 in unit test workflow fix(dependencies): change geopandas version constraint to require exact version 0.14.4
- **dependencies**: change geopandas version constraint to allow any version up to 0.14.4
- **workflow**: downgrade Python version to 3.8 in unit test workflow
- **workflow**: update ONNX Runtime version in unit test workflow
- **predictor**: improve TFLite model loading with better error handling
- **yolopredict**: disable auto install

### Refactor

- **prediction**: simplify ONNX prediction by removing batch processing loop
- **prediction**: streamline prediction functions for batch processing and remove redundant code
- **tests**: comment out deprecated test for PyTorch model predictions
- **test_predict**: update model path definitions to use os.path.join for better compatibility
- **tests**: update model paths in test_predict.py to use relative paths
- **tests**: update model path and clean up commented code in app_test.py

## v0.0.35 (2024-03-27)

### Fix

- **async**: remove async function on sync implementation

## v0.0.34 (2024-03-27)

### Fix

- **tflite-predictions**: added tflite to test cases , add fix for non test functions

## v0.0.33 (2024-03-27)

### Fix

- **main**: orthogonalize in api endpoint fix

## v0.0.32 (2024-03-25)

## v0.0.30 (2023-12-07)
