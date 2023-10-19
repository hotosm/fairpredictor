## fAIr Predictor

Run your fAIr Model Predictions anywhere ! 

## Prerequisites

- Install ```tensorflow-cpu```

## Example on Collab 
```python
# Install 
!pip install fairpredictor

# Import 
from predictor import predict

# Parameters for your predictions 
bbox=[100.56228021333352,13.685230854641182,100.56383321235313,13.685961853747969]
model_path='checkpoint.h5'
zoom_level=20
tms_url='https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}'

# Run your prediction 
my_predictions=predict(bbox,model_path,zoom_level,tms_url)
print(my_predictions)

## Visualize your predictions 

import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.GeoDataFrame.from_features(my_predictions)
gdf.plot()
plt.show()
```

Works on CPU ! Can work on serverless functions, No other dependencies to run predictions 

## Use raster2polygon 

There is another postprocessing option that supports distance threshold between polygon for merging them , If it is useful for you install raster2polygon by : 
```
pip install raster2polygon
```