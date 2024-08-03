import joblib
import supabase
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

URL = "https://caiimowdcnrwheoelnfg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNhaWltb3dkY25yd2hlb2VsbmZnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMTY1MTM0OCwiZXhwIjoyMDM3MjI3MzQ4fQ.CfPM0xL4K1H8pgmmBiFUv0r2Q4OIQv88IL0br0ytn08"

supabase_client = supabase.create_client(URL, SUPABASE_KEY)

def get_data():
    response = supabase_client.table("data").select('*').execute()
    df = pd.DataFrame(response.data)
    return df

def train_and_save_model(df):
    print("Available columns in the DataFrame:")
    print(df.columns.tolist())

    expected_features = ['hour', 'day', 'month', 'year', 'day_of_week', 'distance', 'passenger_count']
    available_features = [col for col in expected_features if col in df.columns]

    if not available_features:
        raise ValueError("None of the expected features are present in the DataFrame")

    print(f"Using available features: {available_features}")

    if 'fare_amount' not in df.columns:
        raise ValueError("Target variable 'fare_amount' is not present in the DataFrame")

    x = df[available_features]
    y = df['fare_amount']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)#coefficient determination

    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")

    # Save the model
    joblib.dump(model, 'model.joblib')
    print("Model saved as model.joblib")

    return model

if __name__ == "__main__":
    df = get_data()
    model = train_and_save_model(df)