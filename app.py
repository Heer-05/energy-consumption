import os
from flask import (
    Flask,
    render_template,
    request,
)
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model   # <-- clean & stable loader

app = Flask(__name__)
app.secret_key = "change_this_secret_key"

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
N_STEPS = 30
N_FEATURES = 7
FEATURE_COLS = ["Power demand", "temp", "dwpt", "rhum", "wdir", "wspd", "pres"]

last_prediction = None


# -------------------------------------------------------------
# Load model & scalers
# -------------------------------------------------------------
MODEL_PATH = "my_hourly_forecaster.h5"       # <-- UPDATED
SCALER_FEATURES_PATH = "scaler_features.pkl"
SCALER_TARGET_PATH = "scaler_target.pkl"

model = None
scaler_features = None
scaler_target = None

try:
    print(f"🔄 Loading model from: {MODEL_PATH}")
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")

    with open(SCALER_FEATURES_PATH, "rb") as f:
        scaler_features = pickle.load(f)
    with open(SCALER_TARGET_PATH, "rb") as f:
        scaler_target = pickle.load(f)

    print("✅ Scalers loaded successfully")
except Exception as e:
    print("❌ Error loading model/scalers:", e)
    model = None


# -------------------------------------------------------------
# Prediction Helper
# -------------------------------------------------------------
def run_prediction_from_window(window_df, temp, dwpt, rhum, wdir, wspd, pres):

    if model is None:
        raise RuntimeError("Model not loaded")

    window = window_df.copy().reset_index(drop=True)
    window = window[FEATURE_COLS].copy()

    # Update last row values
    window.iloc[-1] = [
        window.iloc[-1]["Power demand"],
        float(temp),
        float(dwpt),
        float(rhum),
        float(wdir),
        float(wspd),
        float(pres),
    ]

    # Scale
    scaled_window = scaler_features.transform(window)
    X_predict = scaled_window.reshape(1, N_STEPS, N_FEATURES)

    # Predict
    scaled_pred = model.predict(X_predict)
    final_pred = scaler_target.inverse_transform(scaled_pred)

    return float(final_pred[0][0]), window


# -------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------
@app.route("/")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    global last_prediction

    default_values = {
        "temp": 25.0,
        "dwpt": 18.0,
        "rhum": 50.0,
        "wdir": 180.0,
        "wspd": 10.0,
        "pres": 1013.0,
    }

    prediction_value = None
    chart_data = None
    table_rows = None
    error_message = None

    if request.method == "POST":

        # CSV
        file = request.files.get("csv_file")
        if not file:
            error_message = "Please upload a CSV file."
        else:
            try:
                data = pd.read_csv(file)
            except Exception as e:
                error_message = "Failed to read CSV: " + str(e)

        if error_message is None:
            if len(data) < N_STEPS:
                error_message = "CSV must have at least 30 rows."
            else:
                missing = [c for c in FEATURE_COLS if c not in data.columns]
                if missing:
                    error_message = f"CSV missing required columns: {missing}"

        # Inputs
        def gf(name, d):
            try:
                return float(request.form.get(name, d))
            except:
                return d

        temp = gf("temp", default_values["temp"])
        dwpt = gf("dwpt", default_values["dwpt"])
        rhum = gf("rhum", default_values["rhum"])
        wdir = gf("wdir", default_values["wdir"])
        wspd = gf("wspd", default_values["wspd"])
        pres = gf("pres", default_values["pres"])

        default_values.update(
            temp=temp, dwpt=dwpt, rhum=rhum, wdir=wdir, wspd=wspd, pres=pres
        )

        # Predict
        if error_message is None:
            try:
                window_df = data.tail(N_STEPS)
                pred, modified_window = run_prediction_from_window(
                    window_df, temp, dwpt, rhum, wdir, wspd, pres
                )

                prediction_value = pred

                # Chart
                minutes = list(range(-N_STEPS + 1, 1))
                powers = modified_window["Power demand"].astype(float).tolist()

                chart_data = [
                    {"minute": int(m), "power": float(p)}
                    for m, p in zip(minutes, powers)
                ]
                chart_data.append(
                    {"minute": 1, "power": float(prediction_value)}
                )

                table_rows = modified_window.to_dict(orient="records")

                last_prediction = {
                    "value": prediction_value,
                    "chart_data": chart_data,
                }

            except Exception as e:
                error_message = "Prediction failed: " + str(e)

    return render_template(
        "predict.html",
        active_tab="Prediction",
        defaults=default_values,
        prediction=prediction_value,
        chart_data=chart_data,
        table_rows=table_rows,
        table_columns=FEATURE_COLS,
        error_message=error_message,
    )


@app.route("/guidance")
def guidance():
    from datetime import datetime
    return render_template(
        "guidance.html",
        active_tab="Guidance",
        current_month=datetime.today().month,
    )


@app.route("/awareness")
def awareness():
    return render_template("awareness.html", active_tab="Awareness")


@app.route("/analysis")
def analysis():
    return render_template(
        "analysis.html",
        active_tab="Analysis",
        last_prediction=last_prediction,
    )


@app.route("/about")
def about():
    return render_template("about.html", active_tab="About")


# -------------------------------------------------------------
# Local run
# -------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
