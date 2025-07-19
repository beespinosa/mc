# Bank Marketing Model

This repository contains a complete pipeline for modeling and scoring a bank marketing dataset, structured into two environments:

---

## Directory Structure

- `sandbox/` – **Exploratory** notebooks and visualizations  
  Use this folder for:
  - Feature exploration
  - EDA (exploratory data analysis)
  - Model experimentation

- `prod/` – **Production-ready code**  
  Use this folder for:
  - Final preprocessing logic
  - Model training and scoring
  - FastAPI-based serving

---

## Dataset

We use the **Bank Marketing** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing).

Save the file `bank-full.csv` in the following location:

```
data_input/bank-full.csv
```

Make sure to use `;` as the separator when loading the CSV file.

---

## How to Run

### Local CLI Commands

Run EDA or train the model using:

```bash
python prod/eda.py data_input/bank-full.csv
python prod/main.py data_input/bank-full.csv
```

---

### API Server

Start the FastAPI service:

```bash
uvicorn prod.api:app --reload
```

Then open your browser to:

```
http://localhost:8000/docs
```

This gives you an interactive Swagger UI to test the API.

---

## Example API Input

Here's a valid JSON payload you can POST to `/predict`:

```json
{
  "age": 44,
  "job": "entrepreneur",
  "marital": "married",
  "education": "tertiary",
  "default": "no",
  "balance": 1729,
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "day": 15,
  "month": "october",
  "campaign": 1,
  "pdays": 8,
  "previous": 1,
  "poutcome": "success"
}
```

The API will return:

```json
{
  "upsell_probability": 0.2716
}
```

---

## Environments

The use of two separate Python environments are recommended:

- `requirements-prod.txt` — for API serving and clean deployments
- `requirements-sandbox.txt` — includes dev tools like Jupyter and visualization libraries