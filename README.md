## fAIr Predictor
 
Run your fAIr Model Predictions anywhere ! 

## Example on python 
```python
# Install 
!pip install fairpredictor

import asyncio

from predictor import DEFAULT_OAM_TMS_MOSAIC, DEFAULT_RAMP_MODEL, predict

# Parameters for your predictions
bbox = [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969]
model_path = DEFAULT_RAMP_MODEL
zoom_level = 20
tms_url = DEFAULT_OAM_TMS_MOSAIC

# Run your prediction
my_predictions = asyncio.run(predict(bbox, model_path, zoom_level, tms_url))
print(my_predictions)
```

Works on CPU ! Can work on serverless functions, No other dependencies to run predictions 


## Load Testing

**CAUTION : Always take permission of server admin before you perform load test** 

In order to perform load testing we use Locust , To enable this hit following command within the root dir 

- Install locust

    ```
    pip install locust
    ```

- Run locust script
    ```
    locust -f locust.py
    ```
Populate your HOST and replace it with BASE URL of the Predictor URL 


## Docker 

### Build 
```bash
sudo docker build . -t fairpredictor 
```

### Run 
```bash
sudo docker run --rm --name fairpredictor -v $(pwd):/mnt -p 8000:8000 fairpredictor
```

### Navigate to localhost:8000 and shoot following request body 
```json
{
  "bbox": [100.56228021333352, 13.685230854641182, 100.56383321235313, 13.685961853747969],
  "checkpoint": "/mnt/tests/checkpoints/ramp/checkpoint.tflite",
  "zoom_level": 20,
  "source": "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}"
}
```

