# Taxi Fare Predictor

## Overview
This project is a machine learning-based system for predicting taxi fares. It includes data processing, model training, and a web interface for users to get fare predictions.

## Components

### 1. Data Processing (`data.csv`)
- Fetches data from Supabase storage
- Preprocesses the data, including handling missing values and outliers
- Calculates trip distances using the Haversine formula
- Saves processed data to Supabase database

### 2. Model Training (`train.py`)
- Retrieves processed data from Supabase
- Trains an XGBoost regression model
- Evaluates model performance using RMSE and R-squared
- Saves the trained model as `model.joblib`

### 3. Web Interface (`main.py`)
- Provides a user-friendly interface for fare prediction using Streamlit
- Allows users to input trip details (date, time, distance, passengers)
- Uses the trained model to predict fares

### 4. Data Upload (`upload-s3.py`)
- Uploads the initial dataset (`data.csv`) to Supabase storage

## Setup and Installation

1. Clone the repository
2. Install required packages:
   ```
   pip install supabase pandas numpy scikit-learn xgboost streamlit joblib
   ```
3. Set up Supabase:
   - Create a Supabase project
   - Update the `URL` and `SUPABASE_KEY` in the scripts with your project details

## Usage

1. Run `upload-s3.py` to upload the initial dataset to Supabase storage
2. Execute `pre-processing.py` to process the data and save it to the Supabase database
3. Run `train.py` to train and save the prediction model
4. Launch the web interface:
   ```
   streamlit run main.py
   ```

## Security Note
The Supabase URL and key are currently hardcoded in the scripts. For production use, it's recommended to use environment variables or a secure configuration management system.

## Contributing
Contributions to improve the model accuracy, add features, or enhance the user interface are welcome. Please submit a pull request with your changes.
