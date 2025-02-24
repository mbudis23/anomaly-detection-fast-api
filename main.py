import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import tensorflow as tf
import numpy as np
from typing import List

app = FastAPI()

try:
    ml_isolation_forest = joblib.load("machine_learnings/model.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading Isolation Forest model: {str(e)}")

try:
    ml_lstm_autoencoder = tf.keras.models.load_model("machine_learnings/lstm_autoencoder.keras", compile=False)
except Exception as e:
    raise RuntimeError(f"Error loading LSTM Autoencoder model: {str(e)}")


class InputIsolationForest(BaseModel):
    Total: float
    Count: int
    Maximum: float
    Minimum: float
    Average: float
    Billedsize: float = Field(..., alias="_Billedsize")
    Total_rolling_avg: float
    Total_rolling_stddev: float
    Count_rolling_avg: float
    Count_rolling_stddev: float
    Maximum_rolling_avg: float
    Maximum_rolling_stddev: float
    Minimum_rolling_avg: float
    Minimum_rolling_stddev: float
    Average_rolling_avg: float
    Average_rolling_stddev: float
    Billedsize_rolling_avg: float = Field(..., alias="_Billedsize_rolling_avg")
    Billedsize_rolling_stddev: float = Field(..., alias="_Billedsize_rolling_stddev")
    Year: int
    Month: int
    Day: int
    DayOfWeek: int
    Hour: int
    Minute: int
    IsWeekend: bool
    TimeBin_15min: int
    Total_lag_5min: float
    Total_lag_15min: float
    Total_lag_30min: float
    Total_lag_60min: float
    Count_lag_5min: int
    Count_lag_15min: int
    Count_lag_30min: int
    Count_lag_60min: int
    Maximum_lag_5min: float
    Maximum_lag_15min: float
    Maximum_lag_30min: float
    Maximum_lag_60min: float
    Minimum_lag_5min: float
    Minimum_lag_15min: float
    Minimum_lag_30min: float
    Minimum_lag_60min: float
    Average_lag_5min: float
    Average_lag_15min: float
    Average_lag_30min: float
    Average_lag_60min: float
    Billedsize_lag_5min: float = Field(..., alias="_Billedsize_lag_5min")
    Billedsize_lag_15min: float = Field(..., alias="_Billedsize_lag_15min")
    Billedsize_lag_30min: float = Field(..., alias="_Billedsize_lag_30min")
    Billedsize_lag_60min: float = Field(..., alias="_Billedsize_lag_60min")
    MetricsResourcesId: int


class LSTMInputData(BaseModel):
    Total: float
    Count: int
    Maximum: float
    Minimum: float
    Average: float
    BilledSize: float = Field(..., alias="_BilledSize")
    Total_rolling_avg: float
    Total_rolling_stddev: float
    Count_rolling_avg: float
    Count_rolling_stddev: float
    Maximum_rolling_avg: float
    Maximum_rolling_stddev: float
    Minimum_rolling_avg: float
    Minimum_rolling_stddev: float
    Average_rolling_avg: float
    Average_rolling_stddev: float
    BilledSize_rolling_avg: float = Field(..., alias="_BilledSize_rolling_avg")
    BilledSize_rolling_stddev: float = Field(..., alias="_BilledSize_rolling_stddev")
    Total_lag_5min: float
    Total_lag_15min: float
    Total_lag_30min: float
    Total_lag_60min: float
    Count_lag_5min: int
    Count_lag_15min: int
    Count_lag_30min: int
    Count_lag_60min: int
    Maximum_lag_5min: float
    Maximum_lag_15min: float
    Maximum_lag_30min: float
    Maximum_lag_60min: float
    Minimum_lag_5min: float
    Minimum_lag_15min: float
    Minimum_lag_30min: float
    Minimum_lag_60min: float
    Average_lag_5min: float
    Average_lag_15min: float
    Average_lag_30min: float
    Average_lag_60min: float
    BilledSize_lag_5min: float = Field(..., alias="_BilledSize_lag_5min")
    BilledSize_lag_15min: float = Field(..., alias="_BilledSize_lag_15min")
    BilledSize_lag_30min: float = Field(..., alias="_BilledSize_lag_30min")
    BilledSize_lag_60min: float = Field(..., alias="_BilledSize_lag_60min")
    Allocated_data_storage: float
    AppConnections: int
    Availability: float
    AverageMemoryWorkingSet: float
    AverageResponseTime: float
    browserTimings_networkDuration: float
    browserTimings_processingDuration: float
    browserTimings_receiveDuration: float
    browserTimings_sendDuration: float
    browserTimings_totalDuration: float
    BytesReceived: float
    BytesSent: float
    connection_successful: bool
    cpu_percent: float
    CpuTime: float
    dependencies_duration: float
    dtu_consumption_percent: float
    dtu_limit: float
    dtu_used: float
    exceptions_count: int
    Handles: int
    Http2xx: int
    Http3xx: int
    Http401: int
    Http404: int
    Http4xx: int
    Http5xx: int
    HttpResponseTime: float
    IoOtherBytesPerSecond: float
    IoOtherOperationsPerSecond: float
    IoReadBytesPerSecond: float
    IoReadOperationsPerSecond: float
    IoWriteBytesPerSecond: float
    IoWriteOperationsPerSecond: float
    MemoryWorkingSet: float
    pageViews_count: int
    pageViews_duration: float
    performanceCounters_memoryAvailableBytes: float
    performanceCounters_processCpuPercentage: float
    performanceCounters_processIOBytesPerSecond: float
    performanceCounters_processPrivateBytes: float
    physical_data_read_percent: float
    PrivateBytes: float
    Requests: int
    requests_duration: float
    sessions_count: int
    sessions_percent: float
    storage: float
    storage_percent: float
    Threads: int
    traces_count: int
    workers_percent: float
    xtp_storage_percent: float

@app.post("/predict/if")
def predict_isolation_forest(data: InputIsolationForest):
    try:
        # input_data = np.asarray([[getattr(data, field) for field in data.__annotations__]])
        data_dict = data.model_dump() 
        selected_features = list(data_dict.keys())
        input_data = np.array([data_dict[feature] for feature in selected_features], dtype=np.float32).reshape(1, -1)
        prediction = ml_isolation_forest.predict(input_data)[0]
        if prediction == -1:
            return {"is_anomaly": 1}
        if prediction == 1:
            return {"is_anomaly": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/if2")
def predict_isolation_forest_bulk(data_list: list[InputIsolationForest]):
    try:
        predictions = []
        for data in data_list:
            data_dict = data.model_dump()
            selected_features = list(data_dict.keys())
            input_data = np.array([data_dict[feature] for feature in selected_features], dtype=np.float32).reshape(1, -1)
            prediction = ml_isolation_forest.predict(input_data)[0]
            is_anomaly = 1 if prediction == -1 else 0
            predictions.append({"is_anomaly": is_anomaly})
        
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/lstm")
def predict_lstm(data: LSTMInputData):
    try:
        data_dict = data.model_dump() 
        selected_features = list(data_dict.keys())
        input_array = np.array([data_dict[feature] for feature in selected_features], dtype=np.float32)
        input_array = input_array.reshape(1, 1, 95)
        output = ml_lstm_autoencoder.predict(input_array)
        testMAE = np.mean(np.mean(np.abs(output - input_array), axis=1))
        threshold = 2.0746296834
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
    
@app.post("/predict/lstm2")
def predict_lstm_bulk(data_list: list[LSTMInputData]):
    try:
        predictions = []
        threshold = 2.0746296834
        
        for data in data_list:
            data_dict = data.model_dump()
            selected_features = list(data_dict.keys())
            input_array = np.array([data_dict[feature] for feature in selected_features], dtype=np.float32)
            input_array = input_array.reshape(1, 1, 95)
            output = ml_lstm_autoencoder.predict(input_array)
            testMAE = np.mean(np.mean(np.abs(output - input_array), axis=1))
            is_anomaly = 1 if testMAE > threshold else 0
            predictions.append({"is_anomaly": is_anomaly})
        
        return {"predictions": predictions}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input data: {str(e)}")
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Data processing error: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
