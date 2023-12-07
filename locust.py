from locust import HttpUser, between, task


class MyUser(HttpUser):
    wait_time = between(1, 3)  # Time between requests

    @task
    def predict_api(self):
        payload = {
            "bbox": [
                100.56228021333352,
                13.685230854641182,
                100.56383321235313,
                13.685961853747969,
            ],
            "checkpoint": "/mnt/efsmount/data/trainings/dataset_58/output/training_324/checkpoint.tflite",
            "zoom_level": 20,
            "source": "https://tiles.openaerialmap.org/6501a65c0906de000167e64d/0/6501a65c0906de000167e64e/{z}/{x}/{y}",
            "use_josm_q": "false",
            "merge_adjacent_polygons": "true",
            "confidence": 50,
            "max_angle_change": 15,
            "skew_tolerance": 15,
            "tolerance": 0.5,
            "area_threshold": 3,
            "tile_overlap_distance": 0.15,
        }

        headers = {"Content-Type": "application/json"}

        response = self.client.post("predict/", json=payload, headers=headers)

        # Print response status code and content
        print(f"Response Status Code: {response.status_code}, Content: {response.text}")
