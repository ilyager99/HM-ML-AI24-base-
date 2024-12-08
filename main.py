from fastapi import FastAPI, UploadFile, File
import pickle
from pydantic import BaseModel
from typing import List
import pandas as pd
from io import BytesIO
from joblib import load
import re
from starlette.responses import StreamingResponse
import numpy as np

app = FastAPI()

model = load('ridge_model.joblib')
ohe = pickle.load(open('ohe.pkl', 'rb'))

original_feature_names = ohe.get_feature_names_out()

class DataPreprocessing:
    def __init__(self, data):
        if isinstance(data, dict):
            self.df = pd.DataFrame(data, index=[0])
        else:
            self.df = data

    def extract_number(self, value):
        match = re.search(r'\d+(?:\.\d+)?', str(value))
        return float(match.group()) if match else np.nan
    def data_cleaning(self):
        change_col = ['mileage', 'engine', 'max_power']
        for column in change_col:
            if column in self.df.columns:
                self.df[column] = self.df[column].apply(self.extract_number)

        med = pickle.load(open('median_score.pkl', 'rb'))
        fill = ['mileage', 'engine', 'max_power', 'seats']
        for col in fill:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(med[col])

        if 'engine' in self.df.columns:
            self.df['engine'] = self.df['engine'].astype(int)
        if 'seats' in self.df.columns:
            self.df['seats'] = self.df['seats'].astype(int)
        return self

    def drop_col_torque(self):
        if 'torque' in self.df.columns:
            self.df = self.df.drop(columns=['torque'])
        return self

    def ohe(self):
        categorical_features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
        if all(feature in self.df.columns for feature in categorical_features):
            enc_data = ohe.transform(self.df[categorical_features])
            enc = pd.DataFrame(enc_data.toarray(), columns=ohe.get_feature_names_out(categorical_features))
            self.df = self.df.join(enc)
            self.df.drop(categorical_features, axis=1, inplace=True)

        if 'selling_price' in self.df.columns:
            self.df.drop(columns=['selling_price'], inplace=True)

        self.df = self.df.select_dtypes(include=['int', 'float', 'bool']).copy()
        return self.df

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> str:
    data = item.dict()
    preprocessor = DataPreprocessing(data)
    preprocessor = preprocessor.data_cleaning().drop_col_torque().ohe()

    prediction = model.predict(preprocessor)

    return f'prediction price: {float(prediction[0]):.2f}'

@app.post("/predict_items", response_class=StreamingResponse)
async def predict_items(file: UploadFile = File(...)):
    content = await file.read()
    df_test = pd.read_csv(BytesIO(content))

    required_columns = ['mileage', 'engine', 'max_power', 'fuel', 'seller_type', 'transmission', 'owner', 'seats']
    missing_columns = [col for col in required_columns if col not in df_test.columns]
    if missing_columns:
        return {"error": f"The following required columns are missing: {', '.join(missing_columns)}"}

    if 'selling_price' in df_test.columns:
        df_test.drop(columns=['selling_price'], inplace=True)

    formatting = DataPreprocessing(df_test)
    formatting_data = formatting.data_cleaning().drop_col_torque().ohe()

    predict = model.predict(formatting_data)
    df_test['predicted_price'] = predict

    output_stream = BytesIO()
    df_test.to_csv(output_stream, index=False)
    output_stream.seek(0)

    response = StreamingResponse(output_stream, media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=predictions_output.csv"
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)