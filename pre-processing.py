import supabase
import requests
import pandas as pd
import numpy as np
import io
from datetime import datetime

URL = "https://caiimowdcnrwheoelnfg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNhaWltb3dkY25yd2hlb2VsbmZnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMTY1MTM0OCwiZXhwIjoyMDM3MjI3MzQ4fQ.CfPM0xL4K1H8pgmmBiFUv0r2Q4OIQv88IL0br0ytn08"

supabase_client = supabase.create_client(URL, SUPABASE_KEY)


def get_csv():
    bucket = supabase_client.storage.get_bucket("data")
    url = bucket.get_public_url("data.csv")
    return url

def pre_process_data():
    url = get_csv()
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))

    initial_count = len(df)
    print(f"Initial row count: {initial_count}")

    # Remove the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Generate a new numeric ID
    df['id'] = range(1, len(df) + 1)

    # Convert pickup_datetime to datetime object
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # Clear dates if longitude or latitude is 0
    invalid_coords_count = df[(df['pickup_longitude'] == 0) | (df['pickup_latitude'] == 0) |
                              (df['dropoff_longitude'] == 0) | (df['dropoff_latitude'] == 0)].shape[0]
    df.loc[(df['pickup_longitude'] == 0) | (df['pickup_latitude'] == 0) |
           (df['dropoff_longitude'] == 0) | (df['dropoff_latitude'] == 0), 'pickup_datetime'] = pd.NaT
    
    print(f"Rows with invalid coordinates: {invalid_coords_count} ({invalid_coords_count/initial_count*100:.2f}%)")

    # Extract features from pickup_datetime
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    

    # Handle missing values
    missing_values_count = df.isnull().sum().sum()
    df_clean = df.dropna()
    rows_dropped = len(df) - len(df_clean)
    print(f"Rows dropped due to missing values: {rows_dropped} ({rows_dropped/initial_count*100:.2f}%)")
    df = df_clean

    # Remove extreme outliers in fare_amount
    fare_outliers_count = df[(df['fare_amount'] <= 0) | (df['fare_amount'] >= 1000)].shape[0]
    df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 1000)]
    print(f"Rows dropped due to fare outliers: {fare_outliers_count} ({fare_outliers_count/initial_count*100:.2f}%)")

    # Calculate trip distances
    df['distance'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'],
                                        df['dropoff_latitude'], df['dropoff_longitude'])

    # Select only the required columns for training
    required_columns = ['id', 'fare_amount', 'distance', 'hour', 'day', 'month', 'year', 'day_of_week', 'passenger_count']

    # Ensure all required columns are present
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the dataset")

    # Select only the required columns
    df = df[required_columns]

    final_count = len(df)
    total_dropped = initial_count - final_count
    print(f"Total rows dropped: {total_dropped} ({total_dropped/initial_count*100:.2f}%)")
    print(f"Final row count: {final_count}")

    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth.
    """
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def verify_connection():
    try:
        response = supabase_client.table('data').select('*').limit(1).execute()
        print("Successfully connected to Supabase and verified 'data' table exists")
        return True
    except Exception as e:
        print(f"Error connecting to Supabase or verifying 'data' table: {str(e)}")
        return False

def save_to_db(df):
    # Convert DataFrame to list of dictionaries
    records = df.to_dict('records')

    # Split the data into smaller chunks
    chunk_size = 100  # Adjust this value based on your needs
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        try:
            response = supabase_client.table("data").insert(chunk).execute()
            print(f"Inserted chunk {i // chunk_size + 1} of {len(records) // chunk_size + 1}")
            if hasattr(response, 'error') and response.error:
                print(f"Error inserting chunk: {response.error}")
        except Exception as e:
            print(f"Error inserting chunk: {str(e)}")
            print(f"First record in problematic chunk: {chunk[0]}")

    print("Data saving process completed")

def main():
    if not verify_connection():
        return

    try:
        df = pre_process_data()
        save_to_db(df)
    except Exception as e:
        print(f"An error occurred during data processing or insertion: {str(e)}")

if __name__ == "__main__":
    main()



    