from locust import HttpUser, between, task


class MyUser(HttpUser):
    wait_time = between(1, 3)  # Time between requests

    @task
    def predict_api(self):
        payload = {
            "bbox": [
                85.52111327648163,
                27.632799326007426,
                85.52421927452089,
                27.634308247990774,
            ],
            "model_id": "3",
            "checkpoint": "/mnt/efsmount/data/trainings/dataset_3/output/training_22/checkpoint.tflite",
            "zoom_level": 19,
            "source": "https://tiles.openaerialmap.org/64d3642319cb3a000147a5be/0/64d3642319cb3a000147a5bf/{z}/{x}/{y}",
            "confidence": 90,
            "use_josm_q": False,
            "max_angle_change": 15,
            "skew_tolerance": 15,
            "tolerance": 0.3,
            "area_threshold": 4,
        }
        headers = {"Content-Type": "application/json"}

        response = self.client.post("predict/", json=payload, headers=headers)

        # Print response status code and content
        print(f"Response Status Code: {response.status_code}, Content: {response.text}")
