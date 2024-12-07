import streamlit as st
import pandas as pd
import sqlite3
import os
from datetime import datetime
import joblib
import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objs as go
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# Define the path to the SQLite database
db_path = os.path.join(os.getcwd(), 'db', 'incidents.db')
model = joblib.load('data/threat/xg_boost_model.pkl')
encoder = joblib.load('data/threat/encoder.pkl')

# Function to create a connection to the database
def get_connection():
    conn = sqlite3.connect(db_path)
    return conn

# Function to execute queries and fetch data
def fetch_data(query, params=()):
    conn = get_connection()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

# Function to execute insert, update, delete operations
def execute_query(query, params=()):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)
    conn.commit()
    conn.close()

# Function for threat assessment
def threat_assessment():
    st.subheader("Threat Assessment")
    
    # Input fields for prediction
    neighborhood = st.selectbox("Select Neighborhood", encoder.categories_[0])
    day_of_week = st.selectbox("Select Day of Week", encoder.categories_[1])
    hour = st.number_input("Enter Hour (0-23)", min_value=0, max_value=23, value=12)

    if st.button("Predict Threat Score"):
        # Prepare input data for the model
        input_data = pd.DataFrame([[neighborhood, day_of_week, hour]], columns=['neighborhood', 'day_of_week', 'hour'])
        encoded_input = encoder.transform(input_data[['neighborhood', 'day_of_week']])
        input_final = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())
        input_final['hour'] = hour
        
        # Predict the raw threat score
        raw_threat_score = model.predict(input_final)[0]
        
        # Normalize the threat score to the range 0-100
        max_score = 100  # Set the maximum possible score
        normalized_threat_score = min(raw_threat_score, max_score)  # Cap the score at 100
        
        st.success(f"The predicted threat score is: {normalized_threat_score:.2f}")

        # Categorize the threat level
        threat_category, threat_color = categorize_threat(normalized_threat_score)
        st.markdown(f"### Threat Level: **{threat_category}**")

        # Gauge chart for threat visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=normalized_threat_score,
            title={'text': "Threat Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': threat_color},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # Display messages based on threat level
        if threat_category == "Low Threat":
            st.info("✅ Safe to travel.")
        elif threat_category == "Moderate Threat":
            st.warning("⚠️ Be cautious while traveling.")
        else:
            st.error("❌ Avoid traveling to this area.")
def categorize_threat(threat_score):
    """
    Categorizes the threat score into levels and assigns a color for visualization.

    Args:
    - threat_score (float): The predicted threat score.

    Returns:
    - Tuple[str, str]: A tuple containing the threat level description and the corresponding color.
    """
    if threat_score < 33:
        return "Low Threat", "green"
    elif 33 <= threat_score < 66:
        return "Moderate Threat", "orange"
    else:
        return "High Threat", "red"

def show_crime_forecast():
    st.subheader('Crime Forecast')

    # Get list of neighborhoods
    neighborhoods = fetch_data("SELECT DISTINCT neighborhood FROM incidents WHERE neighborhood IS NOT NULL")
    neighborhoods = neighborhoods['neighborhood'].tolist()
    neighborhoods.append('All Buffalo')  # Add an option for the entire city

    # Create a dropdown to select neighborhood
    selected_neighborhood = st.selectbox('Select a neighborhood', neighborhoods)

    # Load the forecast data
    if selected_neighborhood == 'All Buffalo':
        forecast_data = pd.read_csv('data/forecast/buffalo_crime_forecast.csv')
        title = 'Forecasted Crimes in Buffalo'
    else:
        forecast_data = pd.read_csv(f'data/forecast/forecast_{selected_neighborhood}.csv')
        title = f'Forecasted Crimes in {selected_neighborhood}'

    # Filter data for current date to last date
    today = datetime.today().date()
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds']).dt.date
    filtered_data = forecast_data[forecast_data['ds'] >= today]

    # Create the graph using Plotly
    fig = px.line(filtered_data, x='ds', y='yhat', 
                  title=title,
                  labels={'ds': 'Date', 'yhat': 'Forecasted Number of Crimes'})

    # Add confidence interval
    fig.add_scatter(x=filtered_data['ds'], y=filtered_data['yhat_upper'], 
                    fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Upper Bound')
    fig.add_scatter(x=filtered_data['ds'], y=filtered_data['yhat_lower'], 
                    fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Lower Bound')

    # Update layout to allow zooming out and set default range
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(label="All", step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # Display the graph
    st.plotly_chart(fig)
def crime_risk_analysis():
    st.subheader('Crime Risk Assessment')

    # Load the pre-trained model and encoders
    rf = joblib.load('data/risk/crime_risk_model.pkl')
    le_neighborhood = joblib.load('data/risk/le_neighborhood.pkl')

    # Get list of neighborhoods
    neighborhoods = le_neighborhood.classes_.tolist()

    # Create a dropdown to select neighborhood
    selected_neighborhood = st.selectbox('Select a neighborhood', neighborhoods)

    # Prepare the input data
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))

    # Create all combinations of day and hour
    combinations = [(day, hour) for day in range(7) for hour in range(24)]

    # Prepare input data for prediction
    X_pred = pd.DataFrame(combinations, columns=['day_of_week', 'hour'])
    X_pred['neighborhood'] = le_neighborhood.transform([selected_neighborhood] * len(X_pred))

    # Make predictions
    y_pred_proba = rf.predict_proba(X_pred)

    # Find the highest risk time
    highest_risk_index = y_pred_proba.max(axis=1).argmax()
    highest_risk_day = days[combinations[highest_risk_index][0]]
    highest_risk_hour = combinations[highest_risk_index][1]

    # Display results
    st.write(f"For {selected_neighborhood}:")
    st.write(f"The day with the highest crime risk is: {highest_risk_day}")
    st.write(f"The hour with the highest crime risk is: {highest_risk_hour}:00")

    # Plot heatmap of risk by day and hour
    risk_matrix = y_pred_proba.max(axis=1).reshape(7, 24)
    fig = px.imshow(risk_matrix, 
                    labels=dict(x="Hour of Day", y="Day of Week", color="Risk Level"),
                    x=hours,
                    y=days,
                    title=f"Crime Risk Heatmap for {selected_neighborhood}")
    st.plotly_chart(fig)
import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import sqlite3


# def load_model(neighborhood_name):
#     model_path = f'data/trends/svm_{neighborhood_name}.pkl'
#     if os.path.exists(model_path):
#         return joblib.load(model_path)
#     else:
#         st.error("Model not found for the selected neighborhood.")
#         return None

def buffalo_crime_analysis():
    st.title("Nieghborhood Crime Analysis")
    
    # Fetch data from database
    df = fetch_data("SELECT * FROM incidents")
    
    df['incident_datetime'] = pd.to_datetime(df['incident_datetime'])
    df['hour'] = df['incident_datetime'].dt.hour
    df['day_of_week'] = df['incident_datetime'].dt.day_name()
    df['processed_neighborhood'] = df['neighborhood'].apply(lambda x: x.lower().strip() if isinstance(x, str) else "")

    tab1 = st.tabs(["Crime Prediction"])[0]

    with tab1:
        st.subheader("Neighborhood Crime Analysis & Prediction")
        neighborhoods = sorted(df['processed_neighborhood'].unique())
        selected_neighborhood = st.selectbox("Select Neighborhood:", neighborhoods)

        if selected_neighborhood:
            neighborhood_data = df[df['processed_neighborhood'] == selected_neighborhood]

            if len(neighborhood_data) > 10:
                # Encoding categorical data
                le = LabelEncoder()
                neighborhood_data['day_of_week'] = le.fit_transform(neighborhood_data['day_of_week'])

                # Feature engineering
                X = neighborhood_data[['hour', 'day_of_week', 'month']]  # Include month
                y = neighborhood_data['incident_type_primary']

                # Load pre-trained model and scaler
                model_path = os.path.join('data/trends', f'svm_{selected_neighborhood}.pkl')
                scaler_path = os.path.join('data/trends', f'scaler_{selected_neighborhood}.pkl')

                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    svm_model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)

                    X_scaled = scaler.transform(X)  # Now this should work without error
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Neighborhood Crime Hotspots")
                        st.pydeck_chart(pdk.Deck(
                            map_style='mapbox://styles/mapbox/light-v9',
                            initial_view_state=pdk.ViewState(
                                latitude=neighborhood_data['latitude'].mean(),
                                longitude=neighborhood_data['longitude'].mean(),
                                zoom=14,
                                pitch=45,
                            ),
                            layers=[pdk.Layer(
                                'HexagonLayer',
                                data=neighborhood_data,
                                get_position='[longitude, latitude]',
                                radius=100,
                                elevation_scale=4,
                                elevation_range=[0, 1000],
                                pickable=True,
                                extruded=True,
                            )]
                        ))

                    with col2:
                        st.subheader("Crime Distribution")
                        crime_counts = neighborhood_data['incident_type_primary'].value_counts()
                        fig = px.pie(
                            values=crime_counts.values,
                            names=crime_counts.index,
                            title=f"Crime Types in {selected_neighborhood}"
                        )
                        st.plotly_chart(fig)
                    # Crime prediction section
                    st.subheader("Crime Prediction")
                    
                    # Time input for prediction
                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                    with pred_col1:
                        pred_hour = st.slider("Hour:", 0, 23, 12)
                    with pred_col2:
                        pred_day = st.selectbox("Day:", range(7), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
                    with pred_col3:
                        pred_month = st.selectbox("Month:", range(1, 13))
                    
                    # Prepare the input data for prediction
                    input_data = pd.DataFrame([[pred_hour, pred_day, pred_month]], columns=['hour', 'day_of_week', 'month'])
                    input_scaled = scaler.transform(input_data)  # Scale the input
                    
                    # Make prediction
                    prediction = svm_model.predict_proba(input_scaled)
                    
                    # Show predictions
                    st.subheader("Predicted Crime Risks")
                    risk_cols = st.columns(3)
                    
                    # Get top 3 most likely crime types
                    crime_probs = list(zip(svm_model.classes_, prediction[0]))
                    crime_probs.sort(key=lambda x: x[1], reverse=True)
                    
                    for idx, (crime_type, prob) in enumerate(crime_probs[:3]):
                        with risk_cols[idx]:
                            risk_level = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
                            st.metric(
                                label=f"{crime_type} Risk",
                                value=f"{prob:.1%}",
                                delta=f"{risk_level}",
                                delta_color="inverse" if risk_level == "High" else "normal"
                            )

                else:
                    st.error("Model or scaler for this neighborhood is not available.")
            else:
                st.warning("Insufficient data for this neighborhood. Please select another area.")


def main():
    st.title("Incident Data Management App")

    # Define menu options
    menu = ['Lookup Data', 'Add Data', 'Modify Data', 'Remove Data', 'Threat Assessment','Crime Forecast','Crime Risk Assessment','Neighborhood Crime Trends and Prediction']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Lookup Data':
        st.subheader('Lookup Data')

        # Options for looking up data
        search_options = ['Case Number', 'Incident Type', 'Date Range']
        search_choice = st.selectbox('Search by', search_options)

        if search_choice == 'Case Number':
            case_number = st.text_input('Enter Case Number')
            if st.button('Search'):
                if case_number:
                    query = "SELECT * FROM incidents WHERE case_number = ?"
                    result = fetch_data(query, (case_number,))
                    if not result.empty:
                        # Remove 'location' column if it exists
                        if 'location' in result.columns:
                            result = result.drop(columns=['location'])
                        st.dataframe(result)
                    else:
                        st.warning('No records found for the given case number.')
                else:
                    st.warning('Please enter a case number.')

        elif search_choice == 'Incident Type':
            incident_type = st.text_input('Enter Incident Type')
            if st.button('Search'):
                if incident_type:
                    query = "SELECT * FROM incidents WHERE incident_type_primary LIKE ?"
                    result = fetch_data(query, ('%' + incident_type + '%',))
                    if not result.empty:
                        # Remove 'location' column if it exists
                        if 'location' in result.columns:
                            result = result.drop(columns=['location'])
                        st.dataframe(result)
                    else:
                        st.warning('No records found for the given incident type.')
                else:
                    st.warning('Please enter an incident type.')

        elif search_choice == 'Date Range':
            start_date = st.date_input('Start Date')
            end_date = st.date_input('End Date')
            if st.button('Search'):
                if start_date <= end_date:
                    query = """
                        SELECT * FROM incidents
                        WHERE date(incident_datetime) BETWEEN ? AND ?
                    """
                    result = fetch_data(query, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
                    if not result.empty:
                        # Remove 'location' column if it exists
                        if 'location' in result.columns:
                            result = result.drop(columns=['location'])
                        st.dataframe(result)
                    else:
                        st.warning('No records found for the given date range.')
                else:
                    st.error('Start date must be before or equal to end date.')

    elif choice == 'Add Data':
        st.subheader('Add New Incident')

        # Essential fields
        with st.form('add_incident_form'):
            case_number = st.text_input('Case Number *')
            incident_date = st.date_input('Incident Date *', value=datetime.today())
            incident_time = st.time_input('Incident Time *', value=datetime.now().time())
            incident_type_primary = st.text_input('Incident Type Primary *')
            incident_description = st.text_input('Incident Description *')

            # Optional fields
            parent_incident_type = st.text_input('Parent Incident Type')
            hour_of_day = st.number_input('Hour of Day', min_value=0, max_value=23, step=1, value=datetime.now().hour)
            day_of_week = st.selectbox('Day of Week', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
            address_1 = st.text_input('Address')
            city = st.text_input('City')
            state = st.text_input('State')
            # Removed 'location' input field
            latitude = st.number_input('Latitude', format="%.6f")
            longitude = st.number_input('Longitude', format="%.6f")
            zip_code = st.text_input('Zip Code')
            neighborhood = st.text_input('Neighborhood')
            council_district = st.text_input('Council District')
            council_district_2011 = st.text_input('Council District 2011')
            census_tract = st.number_input('Census Tract', format="%.6f")
            census_block_group = st.text_input('Census Block Group')
            census_block = st.text_input('Census Block')
            census_tract_2010 = st.number_input('Census Tract 2010', format="%.6f")
            census_block_group_2010 = st.text_input('Census Block Group 2010')
            census_block_2010 = st.text_input('Census Block 2010')
            police_district = st.text_input('Police District')
            tractce20 = st.text_input('Tract CE 20')
            geoid20_tract = st.text_input('GEOID20 Tract')
            geoid20_blockgroup = st.text_input('GEOID20 Block Group')
            geoid20_block = st.text_input('GEOID20 Block')
            year = st.number_input('Year', min_value=1900, max_value=2100, value=datetime.now().year)
            month = st.number_input('Month', min_value=1, max_value=12, value=datetime.now().month)
            day = st.number_input('Day', min_value=1, max_value=31, value=datetime.now().day)
            weekday = st.number_input('Weekday', min_value=0, max_value=6, value=datetime.now().weekday())
            hour = st.number_input('Hour', min_value=0, max_value=23, value=datetime.now().hour)

            submitted = st.form_submit_button('Add Incident')

            if submitted:
                # Check if essential fields are filled
                if not case_number or not incident_type_primary or not incident_description:
                    st.error('Please fill in all the required fields marked with *.')
                else:
                    # Combine date and time into a datetime string
                    incident_datetime = datetime.combine(incident_date, incident_time).strftime('%Y-%m-%d %H:%M:%S')

                    # Prepare data for insertion
                    new_incident = {
                        'case_number': case_number,
                        'incident_datetime': incident_datetime,
                        'incident_type_primary': incident_type_primary,
                        'incident_description': incident_description,
                        'parent_incident_type': parent_incident_type or None,
                        'hour_of_day': int(hour_of_day),
                        'day_of_week': day_of_week or None,
                        'address_1': address_1 or None,
                        'city': city or None,
                        'state': state or None,
                        # 'location' field is omitted
                        'latitude': latitude if latitude else None,
                        'longitude': longitude if longitude else None,
                        'zip_code': zip_code or None,
                        'neighborhood': neighborhood or None,
                        'council_district': council_district or None,
                        'council_district_2011': council_district_2011 or None,
                        'census_tract': census_tract if census_tract else None,
                        'census_block_group': census_block_group or None,
                        'census_block': census_block or None,
                        'census_tract_2010': census_tract_2010 if census_tract_2010 else None,
                        'census_block_group_2010': census_block_group_2010 or None,
                        'census_block_2010': census_block_2010 or None,
                        'police_district': police_district or None,
                        'tractce20': tractce20 or None,
                        'geoid20_tract': geoid20_tract or None,
                        'geoid20_blockgroup': geoid20_blockgroup or None,
                        'geoid20_block': geoid20_block or None,
                        'year': int(year) if year else None,
                        'month': int(month) if month else None,
                        'day': int(day) if day else None,
                        'weekday': int(weekday) if weekday else None,
                        'hour': int(hour) if hour else None,
                    }

                    # Check for duplicate case_number
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM incidents WHERE case_number = ?", (case_number,))
                    existing_case = cursor.fetchone()

                    if existing_case:
                        st.error('An incident with this Case Number already exists.')
                    else:
                        # Insert into database
                        columns = ', '.join(new_incident.keys())
                        placeholders = ', '.join(['?'] * len(new_incident))
                        insert_query = f'INSERT INTO incidents ({columns}) VALUES ({placeholders})'
                        try:
                            cursor.execute(insert_query, list(new_incident.values()))
                            conn.commit()
                            st.success('New incident added successfully.')
                        except Exception as e:
                            st.error(f'An error occurred: {e}')
                        finally:
                            conn.close()

    elif choice == 'Modify Data':
        st.subheader('Modify Existing Incident')

        # Input for case_number
        case_number = st.text_input('Enter Case Number of the Incident to Modify')

        if st.button('Fetch Incident'):
            if case_number:
                query = "SELECT * FROM incidents WHERE case_number = ?"
                result = fetch_data(query, (case_number,))
                if not result.empty:
                    incident_data = result.iloc[0]
                    st.session_state['incident_data'] = incident_data.to_dict()
                    st.success('Incident data fetched successfully.')
                else:
                    st.warning('No records found for the given case number.')
            else:
                st.warning('Please enter a case number.')

        if 'incident_data' in st.session_state:
            incident_data = st.session_state['incident_data']
            with st.form('modify_incident_form'):
                # Essential fields (read-only)
                st.text_input('Case Number', value=incident_data['case_number'], disabled=True)
                incident_datetime = st.text_input('Incident Datetime', value=incident_data['incident_datetime'])
                incident_type_primary = st.text_input('Incident Type Primary', value=incident_data['incident_type_primary'])
                incident_description = st.text_area('Incident Description', value=incident_data['incident_description'])

                # Optional fields
                parent_incident_type = st.text_input('Parent Incident Type', value=incident_data.get('parent_incident_type', ''))
                hour_of_day = st.number_input('Hour of Day', min_value=0, max_value=23, value=int(incident_data.get('hour_of_day') or 0), step=1)
                day_of_week = st.text_input('Day of Week', value=incident_data.get('day_of_week', ''))
                address_1 = st.text_input('Address', value=incident_data.get('address_1', ''))
                city = st.text_input('City', value=incident_data.get('city', ''))
                state = st.text_input('State', value=incident_data.get('state', ''))
                # Removed 'location' field
                latitude = st.number_input('Latitude', value=incident_data.get('latitude') or 0.0, format="%.6f")
                longitude = st.number_input('Longitude', value=incident_data.get('longitude') or 0.0, format="%.6f")
                zip_code = st.text_input('Zip Code', value=str(incident_data.get('zip_code', '')))
                neighborhood = st.text_input('Neighborhood', value=incident_data.get('neighborhood', ''))
                council_district = st.text_input('Council District', value=incident_data.get('council_district', ''))
                council_district_2011 = st.text_input('Council District 2011', value=incident_data.get('council_district_2011', ''))
                census_tract = st.number_input('Census Tract', value=incident_data.get('census_tract') or 0.0, format="%.4f")
                census_block_group = st.number_input('Census Block Group', value=incident_data.get('census_block_group') or 0, format="%d")
                census_block = st.number_input('Census Block', value=incident_data.get('census_block') or 0, format="%d")
                census_tract_2010 = st.number_input('Census Tract 2010', value=incident_data.get('census_tract_2010') or 0.0, format="%.4f")
                census_block_group_2010 = st.number_input('Census Block Group 2010', value=incident_data.get('census_block_group_2010') or 0, format="%d")
                census_block_2010 = st.number_input('Census Block 2010', value=incident_data.get('census_block_2010') or 0, format="%d")
                police_district = st.text_input('Police District', value=incident_data.get('police_district', ''))
                tractce20 = st.text_input('Tract CE 20', value=incident_data.get('tractce20', ''))
                geoid20_tract = st.text_input('GEOID20 Tract', value=incident_data.get('geoid20_tract', ''))
                geoid20_blockgroup = st.text_input('GEOID20 Block Group', value=incident_data.get('geoid20_blockgroup', ''))
                geoid20_block = st.text_input('GEOID20 Block', value=incident_data.get('geoid20_block', ''))
                year = st.number_input('Year', value=incident_data.get('year') or 0, format="%d")
                month = st.number_input('Month', min_value=1, max_value=12, value=incident_data.get('month') or 1, format="%d")
                day = st.number_input('Day', min_value=1, max_value=31, value=incident_data.get('day') or 1, format="%d")
                weekday = st.number_input('Weekday', min_value=0, max_value=6, value=incident_data.get('weekday') or 0, format="%d")
                hour = st.number_input('Hour', min_value=0, max_value=23, value=incident_data.get('hour') or 0, format="%d")

                submitted = st.form_submit_button('Update Incident')

                if submitted:
                    # Prepare data for update
                    updated_incident = (
                        incident_datetime,
                        incident_type_primary,
                        incident_description,
                        parent_incident_type or None,
                        int(hour_of_day),
                        day_of_week or None,
                        address_1 or None,
                        city or None,
                        state or None,
                        # 'location' field is omitted
                        latitude if latitude else None,
                        longitude if longitude else None,
                        zip_code or None,
                        neighborhood or None,
                        council_district or None,
                        council_district_2011 or None,
                        census_tract if census_tract else None,
                        int(census_block_group) if census_block_group else None,
                        int(census_block) if census_block else None,
                        census_tract_2010 if census_tract_2010 else None,
                        int(census_block_group_2010) if census_block_group_2010 else None,
                        int(census_block_2010) if census_block_2010 else None,
                        police_district or None,
                        tractce20 or None,
                        geoid20_tract or None,
                        geoid20_blockgroup or None,
                        geoid20_block or None,
                        int(year) if year else None,
                        int(month) if month else None,
                        int(day) if day else None,
                        int(weekday) if weekday else None,
                        int(hour) if hour else None,
                        case_number  # For WHERE clause
                    )

                    update_query = """
                        UPDATE incidents
                        SET incident_datetime = ?, incident_type_primary = ?, incident_description = ?,
                            parent_incident_type = ?, hour_of_day = ?, day_of_week = ?, address_1 = ?, city = ?, state = ?,
                            latitude = ?, longitude = ?, zip_code = ?, neighborhood = ?, council_district = ?, council_district_2011 = ?,
                            census_tract = ?, census_block_group = ?, census_block = ?, census_tract_2010 = ?, census_block_group_2010 = ?,
                            census_block_2010 = ?, police_district = ?, tractce20 = ?, geoid20_tract = ?, geoid20_blockgroup = ?,
                            geoid20_block = ?, year = ?, month = ?, day = ?, weekday = ?, hour = ?
                        WHERE case_number = ?
                    """

                    try:
                        execute_query(update_query, updated_incident)
                        st.success('Incident updated successfully.')
                        del st.session_state['incident_data']
                    except Exception as e:
                        st.error(f'An error occurred: {e}')

    elif choice == 'Remove Data':
        st.subheader('Remove Incident')

        case_number = st.text_input('Enter Case Number of the Incident to Remove')

        if st.button('Delete Incident'):
            if case_number:
                delete_query = "DELETE FROM incidents WHERE case_number = ?"
                try:
                    execute_query(delete_query, (case_number,))
                    st.success('Incident deleted successfully.')
                except Exception as e:
                    st.error(f'An error occurred: {e}')
            else:
                st.warning('Please enter a case number.')

    elif choice == 'Threat Assessment':
        threat_assessment()
    elif choice == 'Crime Forecast':
        show_crime_forecast()
    elif choice == 'Crime Risk Assessment':
        crime_risk_analysis()
    elif choice == 'Neighborhood Crime Trends and Prediction':
        buffalo_crime_analysis()
if __name__ == '__main__':
    main()
