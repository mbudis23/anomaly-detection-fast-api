import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
from typing import List

app = FastAPI()

try:
    ml_isolation_forest = joblib.load("machine_learnings/f3d54393-4ff2-455e-af6e-7c9401782d1a_8489c9b6-51b1-4ea5-8577-7eccdc341832_artifacts/model/model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading Isolation Forest model: {str(e)}")

try:
    ml_lstm_autoencoder = tf.keras.models.load_model("machine_learnings/lstm_autoencoder/lstm_autoencoder.keras", compile=False)
except Exception as e:
    raise RuntimeError(f"Error loading LSTM Autoencoder model: {str(e)}")


class InputIsolationForest(BaseModel):
    Total_AppConnections: float = 0.0
    Count_AppConnections: float = 0.0
    Maximum_AppConnections: float = 0.0
    Minimum_AppConnections: float = 0.0
    Average_AppConnections: float = 0.0
    _BilledSize_AppConnections: float = 0.0
    Total_AppConnections_rolling_avg: float = 0.0
    Total_AppConnections_rolling_stddev: float = 0.0
    Count_AppConnections_rolling_avg: float = 0.0
    Count_AppConnections_rolling_stddev: float = 0.0
    Maximum_AppConnections_rolling_avg: float = 0.0
    Maximum_AppConnections_rolling_stddev: float = 0.0
    Minimum_AppConnections_rolling_avg: float = 0.0
    Minimum_AppConnections_rolling_stddev: float = 0.0
    Average_AppConnections_rolling_avg: float = 0.0
    Average_AppConnections_rolling_stddev: float = 0.0
    _BilledSize_AppConnections_rolling_avg: float = 0.0
    _BilledSize_AppConnections_rolling_stddev: float = 0.0
    Year: int = 0
    Month: int = 0
    Day: int = 0
    DayOfWeek: int = 0
    Hour: int = 0
    Minute: int = 0
    IsWeekend: int = 0
    TimeBin_15min: int = 0
    Total_AppConnections_lag_5min: float = 0.0
    Total_AppConnections_lag_15min: float = 0.0
    Total_AppConnections_lag_30min: float = 0.0
    Total_AppConnections_lag_60min: float = 0.0
    Count_AppConnections_lag_5min: float = 0.0
    Count_AppConnections_lag_15min: float = 0.0
    Count_AppConnections_lag_30min: float = 0.0
    Count_AppConnections_lag_60min: float = 0.0
    Maximum_AppConnections_lag_5min: float = 0.0
    Maximum_AppConnections_lag_15min: float = 0.0
    Maximum_AppConnections_lag_30min: float = 0.0
    Maximum_AppConnections_lag_60min: float = 0.0
    Minimum_AppConnections_lag_5min: float = 0.0
    Minimum_AppConnections_lag_15min: float = 0.0
    Minimum_AppConnections_lag_30min: float = 0.0
    Minimum_AppConnections_lag_60min: float = 0.0
    Average_AppConnections_lag_5min: float = 0.0
    Average_AppConnections_lag_15min: float = 0.0
    Average_AppConnections_lag_30min: float = 0.0
    Average_AppConnections_lag_60min: float = 0.0
    _BilledSize_AppConnections_lag_5min: float = 0.0
    _BilledSize_AppConnections_lag_15min: float = 0.0
    _BilledSize_AppConnections_lag_30min: float = 0.0
    _BilledSize_AppConnections_lag_60min: float = 0.0

class LSTMInputData(BaseModel):
    Total_AppConnections: float
    Count_AppConnections: float
    Maximum_AppConnections: float
    Minimum_AppConnections: float
    Average_AppConnections: float
    BilledSize_AppConnections: float = Field(..., alias="_BilledSize_AppConnections")
    Total_AppConnections_rolling_avg: float
    Total_AppConnections_rolling_stddev: float
    Count_AppConnections_rolling_avg: float
    Count_AppConnections_rolling_stddev: float
    Maximum_AppConnections_rolling_avg: float
    Maximum_AppConnections_rolling_stddev: float
    Minimum_AppConnections_rolling_avg: float
    Minimum_AppConnections_rolling_stddev: float
    Average_AppConnections_rolling_avg: float
    Average_AppConnections_rolling_stddev: float
    BilledSize_AppConnections_rolling_avg: float = Field(..., alias="_BilledSize_AppConnections_rolling_avg")
    BilledSize_AppConnections_rolling_stddev: float = Field(..., alias="_BilledSize_AppConnections_rolling_stddev")
    Total_AppConnections_lag_5min: float
    Total_AppConnections_lag_15min: float
    Total_AppConnections_lag_30min: float
    Total_AppConnections_lag_60min: float
    Count_AppConnections_lag_5min: float
    Count_AppConnections_lag_15min: float
    Count_AppConnections_lag_30min: float
    Count_AppConnections_lag_60min: float
    Maximum_AppConnections_lag_5min: float
    Maximum_AppConnections_lag_15min: float
    Maximum_AppConnections_lag_30min: float
    Maximum_AppConnections_lag_60min: float
    Minimum_AppConnections_lag_5min: float
    Minimum_AppConnections_lag_15min: float
    Minimum_AppConnections_lag_30min: float
    Minimum_AppConnections_lag_60min: float
    Average_AppConnections_lag_5min: float
    Average_AppConnections_lag_15min: float
    Average_AppConnections_lag_30min: float
    Average_AppConnections_lag_60min: float
    BilledSize_AppConnections_lag_5min: float = Field(..., alias="_BilledSize_AppConnections_lag_5min")
    BilledSize_AppConnections_lag_15min: float = Field(..., alias="_BilledSize_AppConnections_lag_15min")
    BilledSize_AppConnections_lag_30min: float = Field(..., alias="_BilledSize_AppConnections_lag_30min")
    BilledSize_AppConnections_lag_60min: float = Field(..., alias="_BilledSize_AppConnections_lag_60min")




class LSTMInputRequest(BaseModel):
    data: List[LSTMInputData]


@app.post("/predict/if")
def predict_isolation_forest(data: InputIsolationForest):
    try:
        # Konversi data ke array
        input_data = np.asarray([[getattr(data, field) for field in data.__annotations__]])
        
        # Prediksi dengan model Isolation Forest
        prediction = ml_isolation_forest.predict(input_data)[0]

        return {"model": "Isolation Forest", "prediction": int(prediction)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/lstm/")
def predict_lstm_autoencoder(request: LSTMInputRequest):
    try:
        model_input_shape = ml_lstm_autoencoder.input_shape
        expected_timesteps = model_input_shape[1]
        expected_features = model_input_shape[2]

        if len(request.data) == 1:
            single_input = np.asarray([[request.data[0].dict().get(field, 0.0) for field in request.data[0].dict().keys()]], dtype=np.float32)

            single_input[single_input == -1.0] = 0.0

            input_data = np.tile(single_input, (expected_timesteps, 1))

        elif len(request.data) == expected_timesteps:
            input_data = np.asarray([[d.dict().get(field, 0.0) for field in d.dict().keys()] for d in request.data], dtype=np.float32)

            input_data[input_data == -1.0] = 0.0

        else:
            raise ValueError(f"Expected 10 timesteps, but got {len(request.data)}.")

        input_data = input_data.reshape((1, expected_timesteps, expected_features))

        reconstructed = ml_lstm_autoencoder.predict(input_data)

        mse = np.mean(np.abs(input_data - reconstructed), axis=(1, 2))

        threshold = 0.02
        anomaly = 1 if mse[0] > threshold else 0

        return {"model": "LSTM Autoencoder", "prediction": anomaly, "mse": mse[0].tolist()}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Input data error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
