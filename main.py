import pandas as pd
import numpy as np
import requests
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mysql.connector
from sqlalchemy import create_engine
import joblib

url = "https://caiimowdcnrwheoelnfg.supabase.co/storage/v1/object/public/data/data.csv"
response = requests.get(url)
response.raise_for_status()
df = pd.read_csv(io.StringIO(response.text))

# Step 3: Do the required pre-processing techniques
def preprocess_data(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df = df.dropna()
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 1000)]
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['time_segment'] = pd.cut(df['hour'], 
                                bins=[0, 6, 12, 18, 24], 
                                labels=['night', 'morning', 'afternoon', 'evening'])
    df['distance'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'],
                                        df['dropoff_latitude'], df['dropoff_longitude'])
    df['vehicle_type'] = pd.cut(df['passenger_count'], 
                                bins=[0, 3, 5, np.inf], 
                                labels=['min', 'mid', 'max'])
    df = df.dropna()
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c
    return distance

df = preprocess_data(df)

# Step 4: Push the cleaned data to RDS server (PostgreSQL) cloud database
def push_to_rds(df, table_name):
    db_url = "postgresql://postgres.caiimowdcnrwheoelnfg:Dina18@&@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(db_url)
    
    try:
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data successfully pushed to {table_name}")
    except Exception as e:
        print(f"Error pushing data to RDS: {e}")

push_to_rds(df, 'taxi_data')

# Step 5: Pull the cleaned data from cloud server
def pull_from_rds(table_name):
    db_url = "postgresql://postgres.caiimowdcnrwheoelnfg:Dina18@&@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    engine = create_engine(db_url)
    
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        print(f"Data successfully pulled from {table_name}")
        return df
    except Exception as e:
        print(f"Error pulling data from RDS: {e}")
        return None

df = pull_from_rds('taxi_data')

# Step 6: Run the machine learning model and save the model
def train_and_save_model(df):
    features = ['hour', 'day', 'month', 'day_of_week', 'distance', 'passenger_count']
    X = df[features]
    y = df['fare_amount']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R-squared Score: {r2:.2f}")
    
    # Save the model
    joblib.dump(model, 'fare_model.joblib')
    print("Model saved as fare_model.joblib")
    
    return model

model = train_and_save_model(df)
