from fastapi import FastAPI
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from pycaret.classification import *
import pickle
import datetime

app = FastAPI(title="Loan Status Prediction App")

grade_dtypes = pd.read_csv('models/grade_dtypes.csv')
subgrade_dtypes = pd.read_csv('models/subgrade_dtypes.csv')


@app.get("/")
@app.get("/home")
def read_home():
    
    return {"message": "System is healthy"}


class ApplicationData(BaseModel):
    date: datetime.date
    loan_amnt: float
    dti: float
    emp_length: str


@app.post("/predict-application-status")
def predict_stroke(data: ApplicationData):
    data_frame = pd.DataFrame(
        data=np.array([i for i in data]).reshape(1, len(data)),
        columns=[
            "date",
            "loan_amnt",
            "dti",
            "emp_length"]
    )
    app_model = load_model('models/app_status_final.pkl')
    prediction = predict_model(app_model, data_frame)['Score']
    return round(float(prediction), 2)


class GradeData(BaseModel):
    for i, j in grade_dtypes.iterrows():
        if grade_dtypes.loc[i, 0] == 'datetime64[ns]':
            grade_dtypes.loc[i, 'index']: datetime.date
        elif grade_dtypes.loc[i, 0] == 'float64':
            grade_dtypes.loc[i, 'index']: float
        elif grade_dtypes.loc[i, 0] == 'object':
            grade_dtypes.loc[i, 'index']: str
        elif grade_dtypes.loc[i, 0] == 'bool':
            grade_dtypes.loc[i, 'index']: bool


@app.post("/predict-grade")
def predict_grade(data: GradeData):
    data_frame = pd.DataFrame(
        data=np.array([i for i in data]).reshape(1, len(data)),
        columns=[i for i in grade_dtypes.index])

    grade_model = load_model('models/grade_final.pkl')
    prediction = predict_model(grade_model, data_frame)['Score']
    return round(float(prediction), 2)


class SubgradeData(BaseModel):
    for i, j in subgrade_dtypes.iterrows():
        if subgrade_dtypes.loc[i, 0] == 'datetime64[ns]':
            subgrade_dtypes.loc[i, 'index']: datetime.date
        elif subgrade_dtypes.loc[i, 0] == 'float64':
            subgrade_dtypes.loc[i, 'index']: float
        elif subgrade_dtypes.loc[i, 0] == 'object':
            subgrade_dtypes.loc[i, 'index']: str
        elif subgrade_dtypes.loc[i, 0] == 'bool':
            subgrade_dtypes.loc[i, 'index']: bool


@app.post("/predict-subgrade")
def predict_subgrade(data: GradeData):
    data_frame = pd.DataFrame(
        data=np.array([i for i in data]).reshape(1, len(data)),
        columns=[i for i in grade_dtypes.index])

    grade_model = load_model('models/subgrade_final.pkl')
    prediction = predict_model(grade_model, data_frame)['Score']
    return round(float(prediction), 2)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)