# ANN Customer Churn Classification

This project uses an Artificial Neural Network (ANN) model to predict whether a bank customer is likely to churn.

The app is built with Streamlit and loads a pre-trained Keras model (`churn_model.h5`) along with preprocessing artifacts (label encoder, one-hot encoder, and scaler).

## Features

- Predict customer churn probability from user input.
- Includes preprocessing pipeline consistency using saved encoders and scaler.
- Simple Streamlit UI for interactive testing.

## Project Structure

- `app.py`: Streamlit application for inference.
- `churn_model.h5`: Trained ANN model.
- `label_encoder_gender.pkl`: Saved label encoder for `Gender`.
- `onehot_encoder_geo.pkl`: Saved one-hot encoder for `Geography`.
- `scaler.pkl`: Saved feature scaler.
- `Churn_Modelling.csv`: Source dataset.
- `requirements.txt`: Python dependencies.

## Requirements

- Python 3.9+
- pip

## Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run the App

```powershell
streamlit run app.py
```

After launch, open the local URL shown in the terminal (usually `http://localhost:8501`).

## Model Inputs

The app collects the following customer attributes:

- Gender
- Geography
- Credit Score
- Age
- Tenure
- Balance
- Estimated Salary
- Number of Products
- Has Credit Card
- Is Active Member

## Output

- Churn probability score between `0.00` and `1.00`
- Final classification using threshold `0.5`:
	- Likely to churn
	- Not likely to churn

## Notes

- Keep model and preprocessing files in the same directory as `app.py`.
- If you retrain the model, regenerate and replace the related `.pkl` preprocessing files to avoid feature mismatch.