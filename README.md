# Iris MLOps Demo

## Purpose

I built this project to deepen my own understanding of the MLOps pipeline and to demonstrate that understanding: training a model, evaluating it honestly, persisting it, serving it through a validated API, containerizing it, testing it, and automating verification with CI.

The model itself is intentionally simple (a standard scikit-learn classifier on the Iris dataset). The point of this project is the infrastructure built around it, not the model.

## What it does

Predicts flower species (setosa, versicolor, virginica) from four measurements (sepal length, sepal width, petal length, petal width) using a RandomForestClassifier, served over a FastAPI REST endpoint.

## Architecture

train.py             trains and evaluates the model, saves it to model.pkl
model.pkl             the trained, persisted model artifact
main.py               FastAPI service that loads model.pkl and serves predictions
test_main.py          automated tests covering the API's behavior
Dockerfile             packages the service into a portable container
.github/workflows/    CI pipeline: installs deps, runs tests, builds the Docker image on every push

## Tech stack

- scikit-learn for model training (RandomForestClassifier)
- FastAPI and Pydantic for API serving and request validation
- joblib for model persistence
- Docker for containerization
- pytest for automated testing
- GitHub Actions for CI (test and build verification on every push)

## Running it locally

**1. Set up the environment**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the model**
```bash
python train.py
```
This trains a RandomForestClassifier on the Iris dataset with an 80/20 split and a fixed random seed for reproducibility, prints test accuracy (around 94.7%), and saves the trained model to model.pkl.

**3. Run the API**
```bash
uvicorn main:app --reload
```
Visit http://localhost:8000/docs for interactive API documentation.

## Running it with Docker

```bash
docker build -t iris-api .
docker run -p 8000:8000 iris-api
```
Same result as running it locally, fully self-contained, no local Python setup required.

## API endpoints

**POST /predict**

Request body:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:
```json
0
```
0 is setosa, 1 is versicolor, 2 is virginica.

**GET /health**

Returns {"status": "ok"}. A lightweight endpoint for uptime and monitoring checks, kept separate from the actual prediction logic.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

Covers a valid prediction request, input validation rejecting malformed data (422), and the health check endpoint.

## CI/CD

Every push to main or master automatically:
1. Checks out the code
2. Sets up Python
3. Installs dependencies
4. Runs the test suite
5. Builds the Docker image

Tests run before the Docker build, so broken code is caught early, before time is spent packaging it.

## Design notes

- random_state is fixed on both the train/test split and the model itself, so training produces the same result every time, on any machine.
- train.py and main.py are kept separate. The API loads a pre-trained artifact instead of retraining on startup, so deploys are fast and don't depend on training infrastructure being available.
- Prediction requests and results are logged to the console and to app.log for traceability.
- The /predict endpoint's error handling path is not currently covered by automated tests, since triggering a genuine model failure isn't straightforward without mocking. The next step here would be mocking model.predict to deliberately raise an exception and verify the endpoint responds with a clean 500 instead of crashing.