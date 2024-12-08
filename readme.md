# Buffalo Crime Analysis Project

## Project Overview

This project analyzes and predicts crime patterns in Buffalo using various approaches, including crime risk prediction, crime forecasting, threat score assessments, and neighborhood crime trend analysis.

## Team Members & Questions

1. **Harshith Nallapu** (UB ID: 50598176, Email: hnallapu@buffalo.edu)
   - Question: Real-Time Crime Risk Prediction: Identifying the Highest Risk Day and Hour
   - Experiment Code: `exp/risk.ipynb`
   - Analysis Location: `exp/risk.ipynb`

2. **Sai Mohith Avula** (UB ID: 50604219, Email: savula4@buffalo.edu)
   - Question: Crime Forecasting Using Facebook Prophet Model
   - Experiment Code: `exp/forecast.ipynb`
   - Analysis Location: `exp/forecast.ipynb`

3. **Rithvik Ramdas** (UB ID: 50608493, Email: rramdas@buffalo.edu)
   - Question: Threat Score Assessment
   - Experiment Code: `exp/threat.ipynb`
   - Analysis Location: `exp/threat.ipynb`

4. **Sanhitha Reddy Manikanti** (UB ID: 50602796, Email: sanhitha@buffalo.edu)
   - Question: Analyzing Neighborhood Crime Trends and Forecasting
   - Experiment Code: `exp/trends.ipynb`
   - Analysis Location: `exp/trends.ipynb`

## Folder Structure

- **data/**: Contains all data files used in the project.
  - `data/forecast/`: Forecasting-related data.
  - `data/processed/`: Processed data files.
  - `data/raw/`: Raw crime data.
  - `data/risk/`: Data related to risk analysis.
  - `data/threat/`: Data related to threat score calculations.
  - `data/trends/`: Data related to neighborhood crime trends.

- **db/**: Contains the database used in the project.
  - `db/incidents.db`: SQLite database with crime data.

- **exp/**: Contains all Jupyter notebooks for analysis and experiments.
  - `exp/forecast.ipynb`: Crime forecasting notebook.
  - `exp/risk.ipynb`: Crime risk prediction notebook.
  - `exp/threat.ipynb`: Threat score assessment notebook.
  - `exp/trends.ipynb`: Neighborhood crime trends notebook.

- **scripts/**: Contains Python scripts for data preparation and processing.
  - `scripts/__pycache__/`: Compiled Python files.
  - `scripts/__init__.py`: Initialization file for the script folder.
  - `scripts/create_database.py`: Script to create the database.
  - `scripts/data_preperation.py`: Data preparation script.

- **app.py**: Streamlit application for presenting the crime analysis dashboard.
- **project_report_phase_3.pdf**: Contains project report for phase 3.

## Building and Running the App

To build and run the application from source code, follow these steps:

1. **Set Up the Environment**
   Ensure you have Python 3.11 or a compatible version installed. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. **Prepare the Data**
   Run the data preparation script to set up the necessary database and data:

   ```
   python scripts/create_database.py
   ```

3. **Execute Jupyter Notebooks**
   Open and run all cells in each of the following notebooks:
   - `exp/forecast.ipynb`
   - `exp/risk.ipynb`
   - `exp/threat.ipynb`
   - `exp/trends.ipynb`

4. **Run the Streamlit App**
   Launch the Streamlit application:

   ```
   streamlit run app.py
   ```

5. **Access the Dashboard**
   The crime data analysis dashboard will be available in your browser, where you can view and explore the results of the analysis, including crime trends, forecasting, and risk predictions.
