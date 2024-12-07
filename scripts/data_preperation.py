# # scripts/data_preparation.py

import requests
import pandas as pd
import numpy as np
import os
from datetime import datetime

def fetch_data():
    # Your existing code
    url = 'https://data.buffalony.gov/resource/d6g9-xbgu.json'
    df_list = []
    offset = 0
    limit = 1000  # Adjust this value as needed
    cutoff_date = datetime(2009, 1, 1)  # Set the cutoff date to the end of 2009

    while True:
        params = {
            '$limit': limit,
            '$offset': offset,
            '$order': 'incident_datetime DESC'  # Sort by incident_datetime in descending order
        }
        response = requests.get(url, params=params)
        data = response.json()
        df_page = pd.DataFrame(data)

        if df_page.empty:
            break

        # Convert incident_datetime to datetime objects
        df_page['incident_datetime'] = pd.to_datetime(df_page['incident_datetime'])

        # Check if we've reached data before or equal to 2009
        if df_page['incident_datetime'].min() <= cutoff_date:
            # Filter out rows after 2009
            df_page = df_page[df_page['incident_datetime'] <= cutoff_date]
            df_list.append(df_page)
            break

        df_list.append(df_page)
        offset += limit

    df = pd.concat(df_list, ignore_index=True)
    return df

def preprocess_data(df):
    # Replace long incident descriptions
    df['incident_description'] = df['incident_description'].str.replace(
        'Buffalo Police are investigating this report of a crime. It is important to note that this is very preliminary information and further investigation as to the facts and circumstances of this report may be necessary.',
        'under investigation',
        regex=False
    )

    # Replace 'UNKNOWN' with NaN
    df = df.replace('UNKNOWN', np.nan)

    # Sort by incident_datetime
    df = df.sort_values(by='incident_datetime')

    # Extract date and time components
    df['year'] = df['incident_datetime'].dt.year
    df['month'] = df['incident_datetime'].dt.month
    df['day'] = df['incident_datetime'].dt.day
    df['weekday'] = df['incident_datetime'].dt.weekday
    df['hour'] = df['incident_datetime'].dt.hour

    # Convert strings to lowercase
    df['incident_type_primary'] = df['incident_type_primary'].str.lower()
    df['parent_incident_type'] = df['parent_incident_type'].str.lower()
    df['address_1'] = df['address_1'].str.lower()

    # Convert latitude and longitude to float
    df['latitude'] = df['latitude'].astype('float64')
    df['longitude'] = df['longitude'].astype('float64')

    # Drop 'created_at' column if it exists
    df.drop(columns=['created_at'], inplace=True, errors='ignore')

    # Drop rows with any NaN values
    df.dropna(axis='index', inplace=True)

    # Categorize incident types into broader crime categories
    crime_categories = {
        'sexual crime': ['other sexual offense', 'sexual assault', 'rape', 'sexual abuse', 'sodomy'],
        'assault crime': ['agg assault on p/officer', 'aggr assault', 'assault'],
        'vehicle crime': ['theft of vehicles', 'uuv', 'theft of vehicle'],
        'theft crimes': ['burglary', 'larceny/theft', 'robbery', 'theft of services', 'theft', 'breaking & entering'],
        'murder crimes': ['crim negligent homicide', 'homicide', 'manslaughter', 'murder'],
    }

    for category, types in crime_categories.items():
        df['incident_type_primary'] = df['incident_type_primary'].replace(types, category)

    return df

def save_data(df):
    # Get the parent directory of the scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    # Define paths
    raw_data_dir = os.path.join(parent_dir, 'data', 'raw')
    processed_data_dir = os.path.join(parent_dir, 'data', 'processed')

    # Ensure directories exist
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # Save raw data
    raw_data_path = os.path.join(raw_data_dir, 'incidents_raw.csv')
    df.to_csv(raw_data_path, index=False)

    # Save processed data
    processed_data_path = os.path.join(processed_data_dir, 'incidents_processed.csv')
    df.to_csv(processed_data_path, index=False)

if __name__ == '__main__':
    df_raw = fetch_data()
    df_processed = preprocess_data(df_raw)
    save_data(df_processed)
    print("Data fetched, preprocessed, and saved successfully.")
