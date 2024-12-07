# scripts/create_database.py

import sys
import os

# Get the directory of the current script (scripts directory)
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the scripts directory to sys.path
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import sqlite3
import pandas as pd
from data_preperation import fetch_data, preprocess_data

def create_db():
    # Fetch and preprocess data
    print("Fetching data...")
    df_raw = fetch_data()
    print("Preprocessing data...")
    df = preprocess_data(df_raw)
    df.drop(columns=['location'], inplace=True)
    # Define the database path
    db_dir = os.path.join(os.path.dirname(script_dir), 'db')
    db_path = os.path.join(db_dir, 'incidents.db')

    # Ensure the 'db' directory exists
    os.makedirs(db_dir, exist_ok=True)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the 'incidents' table with updated data types
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS incidents (
        case_number TEXT PRIMARY KEY,
        incident_datetime TEXT,
        incident_type_primary TEXT,
        incident_description TEXT,
        parent_incident_type TEXT,
        hour_of_day INTEGER,
        day_of_week TEXT,
        address_1 TEXT,
        city TEXT,
        state TEXT,
        location TEXT,
        latitude REAL,
        longitude REAL,
        zip_code INTEGER,
        neighborhood TEXT,
        council_district TEXT,
        council_district_2011 TEXT,
        census_tract REAL,
        census_block_group INTEGER,
        census_block INTEGER,
        census_tract_2010 REAL,
        census_block_group_2010 INTEGER,
        census_block_2010 INTEGER,
        police_district TEXT,
        tractce20 INTEGER,
        geoid20_tract INTEGER,
        geoid20_blockgroup INTEGER,
        geoid20_block INTEGER,
        year INTEGER,
        month INTEGER,
        day INTEGER,
        weekday INTEGER,
        hour INTEGER
    )
    ''')

    # Insert data into the 'incidents' table
    try:
        df.to_sql('incidents', conn, if_exists='replace', index=False)
        print("Data inserted into the 'incidents' table successfully.")
    except Exception as e:
        print(f"An error occurred while inserting data: {e}")
    finally:
        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print("Database and 'incidents' table created and populated successfully.")

if __name__ == '__main__':
    create_db()
