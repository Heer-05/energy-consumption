from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = "change_this_secret_key"  # needed for flash messages

# -------------------------------------------------------------------
# CONSTANTS (same as Streamlit app)
# -------------------------------------------------------------------
N_STEPS = 30
N_FEATURES = 7
FEATURE_COLS = ["Power demand", "temp", "dwpt", "rhum", "wdir", "wspd", "pres"]

# to store last prediction for the Analysis page
last_prediction = None


# -------------------------------------------------------------------
# Load model & scalers (once, at startup)
# -------------------------------------------------------------------
MODEL_PATH = "power_model.keras"
SCALER_FEATURES_PATH = "scaler_features.pkl"
SCALER_TARGET_PATH = "scaler_target.pkl"

try:
    model = load_model(MODEL_PATH)
    with open(SCALER_FEATURES_PATH, "rb") as f:
        scaler_features = pickle.load(f)
    with open(SCALER_TARGET_PATH, "rb") as f:
        scaler_target = pickle.load(f)
    print("✅ Model and scalers loaded.")
except Exception as e:
    print("❌ Error loading model or scalers:", e)
    model = None
    scaler_features = None
    scaler_target = None


# -------------------------------------------------------------------
# Core prediction helper (same logic as Streamlit's run_prediction_from_window)
# -------------------------------------------------------------------
def run_prediction_from_window(window_df, temp, dwpt, rhum, wdir, wspd, pres):
    """
    window_df: DataFrame with at least N_STEPS rows and columns FEATURE_COLS.
    We keep 'Power demand' from last row, but override the 6 weather features.
    """
    window = window_df.copy().reset_index(drop=True)
    window = window[FEATURE_COLS].copy()

    # Modify last row with new environmental values
    last_row = window.iloc[-1].copy()
    last_row["temp"] = float(temp)
    last_row["dwpt"] = float(dwpt)
    last_row["rhum"] = float(rhum)
    last_row["wdir"] = float(wdir)
    last_row["wspd"] = float(wspd)
    last_row["pres"] = float(pres)
    window.iloc[-1] = last_row

    # Scale features and reshape to (1, N_STEPS, N_FEATURES)
    scaled_window = scaler_features.transform(window)
    X_predict = scaled_window.reshape(1, N_STEPS, N_FEATURES)

    # Model prediction + inverse scaling
    scaled_prediction = model.predict(X_predict)
    final_prediction = scaler_target.inverse_transform(scaled_prediction)
    predicted_power = float(final_prediction[0][0])

    return predicted_power, window


# -------------------------------------------------------------------
# NAVIGATION ROUTES
# -------------------------------------------------------------------

@app.route("/")
@app.route("/predict", methods=["GET", "POST"])
def predict():
    global last_prediction

    # defaults for the form fields
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
        # --- get file ---
        uploaded_file = request.files.get("csv_file")
        if not uploaded_file or uploaded_file.filename == "":
            error_message = "Please upload a CSV file."
        else:
            try:
                data = pd.read_csv(uploaded_file)
            except Exception as e:
                error_message = f"Could not read CSV: {e}"
                data = None

        # --- validate data ---
        if error_message is None and data is not None:
            if len(data) < N_STEPS:
                error_message = f"CSV must have at least {N_STEPS} rows. Yours has {len(data)}."
            else:
                missing = [c for c in FEATURE_COLS if c not in data.columns]
                if missing:
                    error_message = f"CSV is missing required columns: {missing}"

        # --- get numeric inputs ---
        def get_float(name, default):
            try:
                return float(request.form.get(name, default))
            except Exception:
                return default

        temp = get_float("temp", default_values["temp"])
        dwpt = get_float("dwpt", default_values["dwpt"])
        rhum = get_float("rhum", default_values["rhum"])
        wdir = get_float("wdir", default_values["wdir"])
        wspd = get_float("wspd", default_values["wspd"])
        pres = get_float("pres", default_values["pres"])

        # update defaults with submitted values (so form keeps them)
        default_values.update(
            {
                "temp": temp,
                "dwpt": dwpt,
                "rhum": rhum,
                "wdir": wdir,
                "wspd": wspd,
                "pres": pres,
            }
        )

        # --- run prediction ---
        if error_message is None:
            if model is None or scaler_features is None or scaler_target is None:
                error_message = (
                    "Model or scalers not loaded. Make sure "
                    "power_model.keras, scaler_features.pkl, scaler_target.pkl are in the app folder."
                )
            else:
                window_df = data.tail(N_STEPS)
                try:
                    pred, modified_window = run_prediction_from_window(
                        window_df, temp, dwpt, rhum, wdir, wspd, pres
                    )
                    prediction_value = pred

                    # build chart data: last 30 power values + predicted next point
                    minutes = list(range(-len(modified_window) + 1, 1))
                    powers = modified_window["Power demand"].astype(float).tolist()
                    chart_data = [
                        {"minute": int(m), "power": float(p)}
                        for m, p in zip(minutes, powers)
                    ]
                    predicted_point = {
                        "minute": minutes[-1] + 1,
                        "power": float(pred),
                    }
                    chart_data.append(predicted_point)

                    # table: last 30 rows of modified window
                    table_rows = modified_window.reset_index(drop=True).to_dict(
                        orient="records"
                    )

                    # store last prediction for Analysis page
                    last_prediction = {
                        "value": prediction_value,
                        "chart_data": chart_data,
                    }

                except Exception as e:
                    error_message = f"Prediction failed: {e}"

    # Render template
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
        current_month=datetime.today().month
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


if __name__ == "__main__":
    # For local run; in production, gunicorn/uwsgi will run app
    app.run(host="0.0.0.0", port=5000, debug=True)
