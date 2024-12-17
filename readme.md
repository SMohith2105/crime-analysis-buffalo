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
# Highlights of Phase 3

## 1. Real-Time Crime Risk Prediction
- **Functionality**: Predicts the highest risk day and hour for crimes using historical crime data.
- **Tools Used**: Random Forest Classifier, integrated with Streamlit for real-time predictions.
- **Location in Code**: exp/risk.ipynb, app.py (Real-time prediction interface)
- **How It Works**: The user selects a neighborhood, day of the week, and hour to predict the crime risk. The system predicts crime risk and displays the results via a heatmap visualization.
- **Visualization**: The heatmap is dynamically generated to visualize crime risks at various times of the day and week.
- **Streamlit Code Location**:
    ```python
    st.write("Heatmap visualization of crime risk.")
    st.map(heatmap_data)
    ```
- **Found in**: app.py where the heatmap visualization is integrated after making predictions in the `real_time_risk_prediction()` function.

## 2. Crime Forecasting Using Facebook Prophet Model
- **Functionality**: Forecasts crime trends over time, identifying potential crime surges in Buffalo neighborhoods.
- **Tools Used**: Facebook Prophet model for time-series forecasting.
- **Location in Code**: exp/forecast.ipynb, app.py (Forecast visualization)
- **How It Works**: Forecasts crime trends for the upcoming year based on historical data, both city-wide and by neighborhood. Visualizes the forecasted crime data as a line chart.
- **Visualization**: The app shows a line chart with the forecasted number of crimes for each neighborhood or the entire city.
- **Streamlit Code Location**:
    ```python
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines'))
    st.plotly_chart(fig)
    ```
- **Found in**: app.py within the `crime_forecasting()` function that integrates the Prophet model and displays the predictions.

## 3. Threat Score Assessment
- **Functionality**: Predicts the safety level (threat score) for specific areas and times based on historical crime data.
- **Tools Used**: XGBoost Regressor for threat score prediction, Streamlit for user interface.
- **Location in Code**: exp/threat.ipynb, app.py (Threat prediction interface)
- **How It Works**: The user selects a neighborhood, day of the week, and hour of the day. The system predicts a threat score and categorizes it into low, moderate, or high risk.
- **Visualization**: A gauge chart displays the threat score in an easy-to-understand format, with ranges indicating the level of threat.
- **Streamlit Code Location**:
    ```python
    threat_score = model.predict(encoded_input)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=threat_score, title={"text": "Threat Score"}, gauge={...}))
    st.plotly_chart(fig)
    ```
- **Found in**: app.py inside the `threat_assessment()` function, where the threat score is calculated and visualized.

## 4. Neighborhood Crime Trends and Forecasting
- **Functionality**: Analyzes and forecasts neighborhood crime trends, helping authorities understand where specific crimes are more likely to occur.
- **Tools Used**: SVM Classifiers, Plotly for data visualization.
- **Location in Code**: exp/trends.ipynb, app.py (Neighborhood crime analysis interface)
- **How It Works**: The app predicts the likelihood of various crime types in a given neighborhood based on temporal features.
- **Visualization**: A 3D crime map displays crime density using the Hexagon Layer, color-coded by crime intensity.
- **Streamlit Code Location**:
    ```python
    fig = go.Figure(go.Scattermapbox(lat=crime_data['lat'], lon=crime_data['lon'], mode='markers', marker=go.scattermapbox.Marker(size=9)))
    st.plotly_chart(fig)
    ```
- **Found in**: app.py in the `neighborhood_crime_analysis()` function, where the neighborhood crime data is visualized on the map.

## 5. 3D Crime Map Visualization in Web Interface
- **Functionality**: Provides an immersive, interactive map for visualizing crime data across Buffalo, highlighting crime hotspots and trends.
- **Tools Used**: Plotly (for creating interactive 3D maps), Pydeck (for advanced map visualizations)
- **How It Works**: The 3D map displays individual crime incidents and crime density in different neighborhoods of Buffalo. It uses the Hexagon Layer to represent crime density, with color-coding to indicate varying levels of crime intensity. High crime density areas are displayed in darker shades, and lower-density areas are shown in lighter shades. A scatter plot layer overlays individual crime incidents on the map, with interactive tooltips providing details about each crime event. The map allows users to zoom in and explore specific areas of interest, helping authorities visualize high-crime zones and allocate resources more efficiently.
- **Visualization**: The 3D crime map enables users to interact with the crime data in a visually immersive way, providing a comprehensive view of crime distribution.
- **Streamlit Code Location**:
    ```python
    fig = go.Figure(go.Scattermapbox(lat=crime_data['lat'], lon=crime_data['lon'], mode='markers', marker=go.scattermapbox.Marker(size=9)))
    st.plotly_chart(fig)
    ```
- **Found in**: app.py within the function that handles neighborhood crime data visualization, where crime incidents are plotted on an interactive map.

## 6. Data Management and Integration
- **Functionality**: Manages crime data stored in an SQLite database and integrates it into the app for analysis and visualization.
- **Tools Used**: SQLite, Pandas for data manipulation, Joblib for model loading.
- **Location in Code**: get_connection(), fetch_data(), execute_query(), model loading in app.py.
- **How It Works**: The SQLite database (incidents.db) stores historical crime data, which is fetched and processed for analysis. Models (e.g., XGBoost, Prophet) are loaded using Joblib to make predictions based on the crime data.
- **Visualization**: Visual outputs from the predictions are integrated into the app, providing clear insights into crime trends and risks. These visualizations are tied to the models used for predictions (heatmap, line chart, gauge chart, etc.).
- **Streamlit Code Location**: Data-fetching and processing code in functions such as `fetch_data()` in app.py. Visualizations are displayed after models are invoked in the respective functions.

## 7. Web Interface for Real-Time Interaction
- **Functionality**: Allows users (law enforcement, city planners) to interact with the crime prediction and analysis models in real-time.
- **Tools Used**: Streamlit for web app development.
- **Location in Code**: app.py
- **How It Works**: Users input parameters like neighborhood, crime type, and time to see real-time predictions and forecasts.
- **Visualization**: Interactive charts (line, pie, bar) and maps (scatter plots, heatmaps) display crime trends, risk scores, and forecasts.
- **Streamlit Code Location**:
    ```python
    fig = px.pie(crime_data, names='crime_type', title='Crime Type Distribution')
    st.plotly_chart(fig)

    fig = px.bar(crime_data, x='day_of_week', y='crime_count', title='Crime Distribution by Day')
    st.plotly_chart(fig)
    ```
- **Found in**: app.py within the functions where visualizations are generated based on user inputs.
