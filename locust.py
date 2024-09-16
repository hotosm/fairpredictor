from locust import HttpUser, between, task


class MyUser(HttpUser):
    wait_time = between(1, 3)  # Time between requests

    @task
    def predict_api(self):
        payload = {
            "bbox": [
                39.73467081522629,
                -5.054039921565167,
                39.73741203283951,
                -5.05203875471145,
            ],
            "model_id": "118",
            "checkpoint": "/mnt/efsmount/data/trainings/dataset_184/output/training_417/checkpoint.tflite",
            "zoom_level": 19,
            "source": "https://tiles.openaerialmap.org/5d971f6ae2b1f300057cb312/0/5d971f6ae2b1f300057cb313/{z}/{x}/{y}",
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
