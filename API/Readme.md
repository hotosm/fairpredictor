## FastAPI Prediction API

Contains a FastAPI-based API for making predictions using a fAIr model. It provides an endpoint to predict results based on specified parameters.

### Prerequisites

- Docker installed on your system

### Getting Started

1. Clone Repo and Navigate to /API

    ```bash
    git clone <this-repository-url>
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

    Go to : ```Localhost:8080/redoc```