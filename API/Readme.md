## FastAPI Prediction API

Contains a FastAPI-based API for making predictions using a fAIr model. It provides an endpoint to predict results based on specified parameters.

### Prerequisites

- Docker installed on your system

### Getting Started

1. Clone Repo and Navigate to /API

    ```bash
    git clone https://github.com/kshitijrajsharma/fairpredictor.git
    cd API
    ```

2. Build Docker Image

    ```bash
    docker build -t predictor-api .
    ```

3. Run Docker Container

    ```bash
    docker run -p 8080:8000 predictor-api
    ```

4. API Documentation

    - Redocly Documentation - > Go to your_API_url/redoc : for eg [localhost:redoc](http://localhost:8080/redoc)
    - Swagger Documentation - > Go to your_API_url/docs : for eg [localhost:docs](http://localhost:8080/docs#/default/predict_api_predict__post)
