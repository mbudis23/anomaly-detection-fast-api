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
    MetricsCode : int  = 0

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
        input_data = np.asarray([[getattr(data, field) for field in data.__annotations__]])
        prediction = ml_isolation_forest.predict(input_data)[0]
        if prediction == -1:
            return {"is_anomaly": 1}
        if prediction == 1:
            return {"is_anomaly": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/lstm")
def predict_lstm(data: LSTMInputData):
    try:
        data_dict = data.model_dump() 
        selected_features = list(data_dict.keys())
        input_array = np.array([data_dict[feature] for feature in selected_features], dtype=np.float32)
        input_array = input_array.reshape(1, 1, 42)
        output = ml_lstm_autoencoder.predict(input_array)
        testMAE = np.mean(np.mean(np.abs(output - input_array), axis=1))
        threshold = 0.439664205838787
        if testMAE > threshold:
            return {'is_anomaly': 1}
        else :
            return {'is_anomaly': 0}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input data: {str(e)}")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")