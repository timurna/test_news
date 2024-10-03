import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import os

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Ensure 'authenticated' and login keys are initialized in session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def authenticate(username, password):
    try:
        # Retrieve the credentials from st.secrets (stored in Streamlit Cloud's secret management)
        stored_username = st.secrets["credentials"]["username"]
        stored_password = st.secrets["credentials"]["password"]
    except KeyError as e:  # Ensure this line is aligned with the try block
        # If the credentials are not found, display an error message
        st.error(f"Error: {e}. Credentials not found in Streamlit secrets.")
        return False
    
    # Compare provided credentials with the stored ones
    return username == stored_username and password == stored_password

def login():
    # Capture user input for username and password
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    
    if st.button("Login", key="login_button"):
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.success("Login successful!")
            st.experimental_rerun()  # Reload the app after successful login
        else:
            st.error("Invalid username or password")

# Ensure proper authentication
if not st.session_state.authenticated:
    login()
else:
    # The user has logged in, continue with the main app
    st.write("Welcome! You are logged in.")

    # Function to apply custom CSS for mobile responsiveness
    def set_mobile_css():
        st.markdown(
            """
            <style>
            @media only screen and (max-width: 600px) {
                .stApp {
                    padding: 0 10px;
                }
                .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
                    font-size: 1.2em !important;
                }
                .headline {
                    font-size: 1.5em !important;
                }
                .stDataFrame th, .stDataFrame td {
                    font-size: 0.8em !important;
                }
                .css-12w0qpk, .css-15tx938, .stSelectbox label, .stTable th, .stTable thead th, .dataframe th {
                    font-size: 0.8em !important;
                }
            }
            .tooltip {
                position: relative;
                display: inline-block;
                border-bottom: 1px dotted black;
            }
            .tooltip .tooltiptext {
                visibility: hidden;
                width: 120px;
                background-color: black;
                color: #fff;
                text-align: center;
                border-radius: 6px;
                padding: 5px 0;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -60px;
                opacity: 0;
                transition: opacity 0.3s;
            }
            .tooltip:hover .tooltiptext {
                visibility: visible;
                opacity: 1;
            }
            </style>
            """, unsafe_allow_html=True
        )

    # Function to download and load the file from Google Drive
    @st.cache
    def download_and_load_data(url):
        # Define the file path for the downloaded parquet file
        parquet_file = '/tmp/newupclean3.parquet'
        
        # Download the file using requests
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises HTTPError for bad responses
            with open(parquet_file, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading file: {e}")
            return None
        
        # Load the parquet file using pandas
        try:
            data = pd.read_parquet(parquet_file)
            data['DOB'] = pd.to_datetime(data['DOB'])
            data['Date'] = pd.to_datetime(data['Date'])
            return data
        except Exception as e:
            st.error(f"Error reading parquet file: {e}")
            return None

    # Google Drive direct download link
    file_url = 'https://drive.google.com/uc?export=download&id=1NNRDRxUR9NjPbNjQtH0qwJNJjP2w_wEl'

    # Only load the dataset **after** successful login
    data = download_and_load_data(file_url)

    if data is None:
        st.error("Failed to load data")
    else:
        # Glossary content and the rest of your logic for displaying tables, metrics, etc.
        set_mobile_css()
        # Continue with the rest of your app logic...

if data is None:
    st.error("Failed to load data")
else:
    # Glossary content with metrics integrated
    glossary = {
        'Score Metrics': '',  
        'Overall Score': 'Player\'s overall performance across all metrics.',
        'Defensive Score': 'Player\'s overall defensive performance. Metrics: TcklMade%, TcklAtt, Tckl, AdjTckl, TcklA3, Blocks, Int, AdjInt, Clrnce',
        'Goal Threat Score': 'Player\'s threat to score goals. Metrics: Goal, Shot/Goal, MinPerGoal, ExpG, xGOT, xG +/-, Shot, SOG, Shot conversion, OnTarget%',
        'Offensive Score': 'Player\'s overall offensive performance. Metrics: 2ndAst, Ast, ExpG, ExpGExPn, Goal, GoalExPn, KeyPass, MinPerChnc, MinPerGoal, PsAtt, PsCmp, Pass%, PsIntoA3rd, PsRec, ProgCarry, ProgPass, Shot, Shot conversion, Shot/Goal, SOG, OnTarget%, Success1v1, Take on into the Box, TakeOn, ThrghBalls, TouchOpBox, Touches, xA, xA +/-, xG +/-, xGOT',
        'Physical Offensive Score': 'Player\'s physical contributions to offensive play. Metrics: PSV-99, Distance, M/min, HSR Distance, HSR Count, Sprint Distance, Sprint Count, HI Distance, HI Count, Medium Acceleration Count, High Acceleration Count, Medium Deceleration Count, High Deceleration Count',
        'Physical Defensive Score': 'Player\'s physical contributions to defensive play. Metrics: Distance OTIP, M/min OTIP, HSR Distance OTIP, HSR Count OTIP, Sprint Distance OTIP, Sprint Count OTIP, HI Distance OTIP, HI Count OTIP, Medium Acceleration Count OTIP, High Acceleration Count OTIP, Medium Deceleration Count OTIP, High Deceleration Count OTIP',
        'Offensive Metrics': '',  
        '2ndAst': 'The pass that assists the assist leading to a goal.',
        'Ast': 'Assists',
        'ExpG': 'Expected goals.',
        'ExpGExPn': 'Expected goals excluding penalties.',
        'Goal': 'Goals scored.',
        'GoalExPn': 'Goals excluding penalties.',
        'KeyPass': 'Passes that directly lead to a shot on goal.',
        'MinPerChnc': 'Minutes per chance created.',
        'MinPerGoal': 'Minutes per goal.',
        'OnTarget%': 'Percentage of shots on target out of total shots.',
        'PsAtt': 'Passes attempted.',
        'PsCmp': 'Passes completed.',
        'Pass%': 'Percentage of completed passes out of total passes attempted.',
        'PsIntoA3rd': 'Passes into the attacking third.',
        'PsRec': 'Passes received by the player.',
        'ProgCarry': 'Progressive carries, advancing the ball significantly.',
        'ProgPass': 'Progressive passes, advancing the ball significantly.',
        'Shot': 'Total shots taken.',
        'Shot conversion': 'Shots on target per goal.',
        'Shot/Goal': 'Total shots per goal.',
        'SOG': 'Shots on goal.',
        'Success1v1': 'Successful dribbles against an opponent.',
        'Take on into the Box': 'Number of successful dribbles into the penalty box.',
        'TakeOn': 'Attempted dribbles against an opponent.',
        'ThrghBalls': 'Through balls played.',
        'TouchOpBox': 'Number of touches in the opponent\'s penalty box.',
        'Touches': 'Total number of touches.',
        'xA': 'Expected assists.',
        'xA +/-': 'Expected Assists +/- difference.',
        'xG +/-': 'Expected goals +/- difference.',
        'xGOT': 'Expected goals on target.',
        'Defensive Metrics': '',  
        'AdjInt': 'Adjusted interceptions, considering context.',
        'AdjTckl': 'Adjusted tackles, considering context.',
        'Blocks': 'Total blocks made.',
        'Clrnce': 'Clearances made.',
        'Int': 'Interceptions made.',
        'Tckl': 'Tackles made.',
        'TcklMade%': 'Percentage of tackles successfully made out of total tackle attempts.',
        'TcklA3': 'Tackles made in the attacking third.',
        'TcklAtt': 'Tackles attempted.',
        'Physical Metrics': '',  
        'PSV-99': 'Peak Sprint Velocity (Maximum Speed).',
        'Distance': 'Total distance covered by the player during the match.',
        'Distance OTIP': 'Distance covered while opponent has ball possession (OTIP).',
        'HI Count': 'High-intensity actions performed.',
        'HI Count OTIP': 'High-intensity actions performed while opponent has ball possession (OTIP).',
        'HI Distance': 'High-intensity distance covered.',
        'HI Distance OTIP': 'High-intensity distance covered while opponent has ball possession (OTIP).',
        'High Acceleration Count': 'High-intensity accelerations performed.',
        'High Acceleration Count OTIP': 'High-intensity accelerations performed while opponent has ball possession (OTIP).',
        'High Deceleration Count': 'High-intensity decelerations performed.',
        'High Deceleration Count OTIP': 'High-intensity decelerations performed while opponent has ball possession (OTIP).',
        'HSR Count': 'Count of high-speed running actions.',
        'HSR Count OTIP': 'High-speed running actions performed while opponent has ball possession (OTIP).',
        'HSR Distance': 'High-speed running distance covered.',
        'HSR Distance OTIP': 'High-speed running distance covered while opponent has ball possession (OTIP).',
        'M/min': 'Meters covered per minute by the player.',
        'M/min OTIP': 'Meters per minute covered while opponent has ball possession (OTIP).',
        'Medium Acceleration Count': 'Medium-intensity accelerations performed.',
        'Medium Acceleration Count OTIP': 'Medium-intensity accelerations performed while opponent has ball possession (OTIP).',
        'Medium Deceleration Count': 'Medium-intensity decelerations performed.',
        'Medium Deceleration Count OTIP': 'Medium-intensity decelerations performed while opponent has ball possession (OTIP).',
        'Sprint Count': 'Total sprints performed.',
        'Sprint Count OTIP': 'Sprint actions performed while opponent has ball possession (OTIP).',
        'Sprint Distance': 'Total distance covered while sprinting.',
        'Sprint Distance OTIP': 'Sprint distance covered while opponent has ball possession (OTIP).'
    }

    # Calculate age from birthdate
    data['DOB'] = pd.to_datetime(data['DOB'])
    today = datetime.today()
    data['Age'] = data['DOB'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))

    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Define position groups with potential overlaps
    position_groups = {
        'IV': ['Left Centre Back', 'Right Centre Back', 'Central Defender'],
        'AV': ['Left Back', 'Right Back'],
        'FLV': ['Left Wing Back', 'Right Wing Back'],
        'AVFLV': ['Left Back', 'Right Back', 'Left Wing Back', 'Right Wing Back'],
        'ZDM': ['Defensive Midfielder'],
        'ZDMZM': ['Defensive Midfielder', 'Central Midfielder'],
        'ZM': ['Central Midfielder'],
        'ZOM': ['Centre Attacking Midfielder'],
        'ZMZOM': ['Central Midfielder', 'Centre Attacking Midfielder'],
        'FS': ['Left Midfielder', 'Right Midfielder', 'Left Attacking Midfielder', 'Right Attacking Midfielder'],
        'ST': ['Left Winger', 'Right Winger', 'Second Striker', 'Centre Forward']
    }

    # Assign positions to multiple groups
    data['Position Groups'] = data['Position_x'].apply(lambda pos: [group for group, positions in position_groups.items() if pos in positions])

    # Convert text-based numbers to numeric, handling percentage metrics
    percentage_metrics = ['TcklMade%', 'Pass%', 'OnTarget%']

    # Remove percentage signs and convert to numeric
    for metric in percentage_metrics:
        if metric in data.columns:
            data[metric] = pd.to_numeric(data[metric].astype(str).str.replace('%', ''), errors='coerce')

    # Convert other text-based numbers to numeric
    physical_metrics = ['PSV-99', 'Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance', 'Sprint Count',
                        'HI Distance', 'HI Count', 'Medium Acceleration Count', 'High Acceleration Count',
                        'Medium Deceleration Count', 'High Deceleration Count', 'Distance OTIP', 'M/min OTIP',
                        'HSR Distance OTIP', 'HSR Count OTIP', 'Sprint Distance OTIP', 'Sprint Count OTIP',
                        'HI Distance OTIP', 'HI Count OTIP', 'Medium Acceleration Count OTIP',
                        'High Acceleration Count OTIP', 'Medium Deceleration Count OTIP', 'High Deceleration Count OTIP']

    offensive_metrics = [
        '2ndAst', 'Ast', 'ExpG', 'ExpGExPn', 'Goal', 'GoalExPn', 'KeyPass', 
        'MinPerChnc', 'MinPerGoal', 'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 
        'PsRec', 'ProgCarry', 'ProgPass', 'Shot', 'Shot conversion', 
        'Shot/Goal', 'SOG', 'OnTarget%', 'Success1v1', 'Take on into the Box', 
        'TakeOn', 'ThrghBalls', 'TouchOpBox', 'Touches', 'xA', 
        'xA +/-', 'xG +/-', 'xGOT'
    ]

    defensive_metrics = [
        'TcklMade%', 'TcklAtt', 'Tckl', 'AdjTckl', 'TcklA3', 'Blocks', 'Int', 'AdjInt', 'Clrnce'
    ]

    goal_threat_metrics = [
        'Goal', 'Shot/Goal', 'MinPerGoal', 'ExpG', 'xGOT', 'xG +/-', 
        'Shot', 'SOG', 'Shot conversion', 'OnTarget%'
    ]

    for metric in physical_metrics + offensive_metrics + defensive_metrics + goal_threat_metrics:
        if metric in data.columns and metric not in percentage_metrics:  # Exclude percentage metrics already processed
            data[metric] = pd.to_numeric(data[metric].astype(str).str.replace(',', '.'), errors='coerce')

    # Fill NaN values with 0 only for players who have any non-NaN value in the group of metrics
    def fill_na_conditionally(df, metric_group):
        # Create a mask where any metric in the group is not NaN
        mask = df[metric_group].notna().any(axis=1)
        # Apply filling only to rows where the mask is True
        df.loc[mask, metric_group] = df.loc[mask, metric_group].fillna(0)

    fill_na_conditionally(data, physical_metrics)
    fill_na_conditionally(data, offensive_metrics)
    fill_na_conditionally(data, defensive_metrics)
    fill_na_conditionally(data, goal_threat_metrics)

    # Initialize the scalers
    scaler = MinMaxScaler(feature_range=(0, 10))
    quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=0)

    # Define physical metrics subsets
    physical_offensive_metrics = [
        'PSV-99', 'Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance', 
        'Sprint Count', 'HI Distance', 'HI Count', 
        'Medium Acceleration Count', 'High Acceleration Count', 
        'Medium Deceleration Count', 'High Deceleration Count'
    ]

    physical_defensive_metrics = [
        'Distance OTIP', 'M/min OTIP', 'HSR Distance OTIP', 'HSR Count OTIP', 
        'Sprint Distance OTIP', 'Sprint Count OTIP', 'HI Distance OTIP', 
        'HI Count OTIP', 'Medium Acceleration Count OTIP', 
        'High Acceleration Count OTIP', 'Medium Deceleration Count OTIP', 
        'High Deceleration Count OTIP'
    ]

    # Calculate the scores
    data['Physical Offensive Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[physical_offensive_metrics].fillna(0))
    ).mean(axis=1)

    data['Physical Defensive Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[physical_defensive_metrics].fillna(0))
    ).mean(axis=1)

    data['Offensive Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[offensive_metrics].fillna(0))
    ).mean(axis=1)

    data['Defensive Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[defensive_metrics].fillna(0))
    ).mean(axis=1)

    data['Goal Threat Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[goal_threat_metrics].fillna(0))
    ).mean(axis=1)

    # **Add the Overall Score by combining all metrics**
    # Create a list of all metrics used in the scores
    all_metrics = list(set(
        physical_offensive_metrics + physical_defensive_metrics + 
        offensive_metrics + defensive_metrics + goal_threat_metrics
    ))

    data['Overall Score'] = scaler.fit_transform(
        quantile_transformer.fit_transform(data[all_metrics].fillna(0))
    ).mean(axis=1)

    # **Calculate Cumulative Averages for Metrics**

    # Ensure the data is sorted
    data = data.sort_values(['League', 'playerFullName', 'Date'])

    # Create a list of metrics for which we want cumulative averages
    metrics_for_cum_avg = ['Overall Score', 'Offensive Score', 'Defensive Score', 
                           'Physical Offensive Score', 'Physical Defensive Score', 'Goal Threat Score'] + \
                           physical_offensive_metrics + physical_defensive_metrics + offensive_metrics + defensive_metrics

    # Remove duplicates
    metrics_for_cum_avg = list(set(metrics_for_cum_avg))

    # Calculate cumulative averages for each player in each league
    for metric in metrics_for_cum_avg:
        data[f'{metric}_cum_avg'] = data.groupby(['League', 'playerFullName'])[metric].expanding().mean().reset_index(level=[0,1], drop=True)

    # Ensure 'authenticated' is in session state before anything else
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    def authenticate(username, password):
        try:
            # Attempt to retrieve the credentials from st.secrets
            stored_username = st.secrets["credentials"]["username"]
            stored_password = st.secrets["credentials"]["password"]
        except KeyError as e:
            # If the credentials are not found, display an error message
            st.error(f"KeyError: {e}. Credentials not found in secrets. Please check your secrets configuration.")
            return False
        
        # Compare provided credentials with the stored ones
        return username == stored_username and password == stored_password

    def login():
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_button"):
            if authenticate(username, password):
                st.session_state.authenticated = True
            else:
                st.error("Invalid username or password")

    if not st.session_state.authenticated:
        login()
    else:
        set_mobile_css()

        # Display the logo at the top
        st.image('logo.png', use_column_width=False, width=800)

        # Create a single row for all the filters
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                leagues = sorted(data['League'].unique())  # Sort leagues alphabetically
                selected_league = st.selectbox("Select League", leagues, key="select_league")

            with col2:
                league_data = data[data['League'] == selected_league]

                # Week Summary and Matchday Filtering Logic
                week_summary = league_data.groupby(['League', 'Week']).agg({'Date': ['min', 'max']}).reset_index()
                week_summary.columns = ['League', 'Week', 'min', 'max']

                week_summary['min'] = pd.to_datetime(week_summary['min'])
                week_summary['max'] = pd.to_datetime(week_summary['max'])

                week_summary['Matchday'] = week_summary.apply(
                    lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})", axis=1
                )

                filtered_weeks = week_summary[week_summary['League'] == selected_league].sort_values(by='min').drop_duplicates(subset=['Week'])

                matchday_options = filtered_weeks['Matchday'].tolist()
                selected_matchday = st.selectbox("Select Matchday", matchday_options, key="select_matchday")

                selected_week = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['Week'].values[0]

                # Get the date of the selected matchday
                selected_date = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['max'].values[0]

            with col3:
                position_group_options = list(position_groups.keys())
                selected_position_group = st.selectbox("Select Position Group", position_group_options, key="select_position_group")

        # Filter the data by the selected position group and up to the selected matchday
        league_and_position_data = data[
            (data['League'] == selected_league) &
            (data['Date'] <= selected_date) &
            (data['Position Groups'].apply(lambda groups: selected_position_group in groups))
        ]

        # Use a container to make the expandable sections span the full width
        with st.container():
            tooltip_headers = {metric: glossary.get(metric, '') for metric in ['Overall Score', 'Offensive Score', 'Defensive Score', 'Physical Offensive Score', 'Physical Defensive Score', 'Goal Threat Score'] + physical_metrics + offensive_metrics + defensive_metrics}

            def display_metric_tables(metrics_list, title):
                with st.expander(title, expanded=False):  # Setting expanded=False to keep it closed by default
                    for metric in metrics_list:
                        if metric not in league_and_position_data.columns:
                            st.write(f"Metric {metric} not found in the data")
                            continue

                        # Get the latest data for each player up to the selected date
                        latest_data = league_and_position_data.sort_values(['playerFullName', 'Date']).groupby('playerFullName').last().reset_index()

                        # Round the Age column to ensure no decimals
                        latest_data['Age'] = latest_data['Age'].round(0).astype(int)

                        # Prepare the data
                        top10 = latest_data[['playerFullName', 'Age', 'newestTeam', 'Position_x', metric, f'{metric}_cum_avg']].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

                        if top10.empty:
                            st.header(f"Top 10 Players in {metric}")
                            st.write("No data available")
                        else:
                            # Reset the index to create a rank column starting from 1
                            top10.reset_index(drop=True, inplace=True)
                            top10.index += 1
                            top10.index.name = 'Rank'

                            # Ensure the Rank column is part of the DataFrame before styling
                            top10 = top10.reset_index()

                            st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)
                            top10.rename(columns={'playerFullName': 'Player', 'newestTeam': 'Team', 'Position_x': 'Position'}, inplace=True)

                            # Format the metric value with cumulative average
                            top10[metric] = top10.apply(
                                lambda row: f"{row[metric]:.2f} ({row[f'{metric}_cum_avg']:.2f})" if pd.notnull(row[f'{metric}_cum_avg']) else f"{row[metric]:.2f}",
                                axis=1
                            )

                            # Remove the cumulative average column from the DataFrame as it's now included in the metric column
                            top10.drop(columns=[f'{metric}_cum_avg'], inplace=True)

                            def color_row(row):
                                return ['background-color: #d4edda' if row['Age'] < 24 else '' for _ in row]

                            top10_styled = top10.style.apply(color_row, axis=1)
                            top10_html = top10_styled.to_html()

                            for header, tooltip in tooltip_headers.items():
                                if tooltip:
                                    top10_html = top10_html.replace(f'>{header}<', f'><span class="tooltip">{header}<span class="tooltiptext">{tooltip}</span></span><')

                            st.write(top10_html, unsafe_allow_html=True)

            display_metric_tables(['Overall Score', 'Offensive Score', 'Goal Threat Score', 'Defensive Score', 'Physical Offensive Score', 'Physical Defensive Score'], "Score Metrics")
            display_metric_tables(physical_offensive_metrics, "Physical Offensive Metrics")
            display_metric_tables(physical_defensive_metrics, "Physical Defensive Metrics")
            display_metric_tables(offensive_metrics, "Offensive Metrics")
            display_metric_tables(defensive_metrics, "Defensive Metrics")

        # Glossary section - Render only after authentication inside an expander
        with st.expander("Glossary"):
            sections = {
                "Score Metrics": [
                    'Overall Score', 'Defensive Score', 'Goal Threat Score', 'Offensive Score', 
                    'Physical Defensive Score', 'Physical Offensive Score'
                ],
                "Offensive Metrics": [
                    '2ndAst', 'Ast', 'ExpG', 'ExpGExPn', 'Goal', 'GoalExPn', 'KeyPass', 
                    'MinPerChnc', 'MinPerGoal', 'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 
                    'PsRec', 'ProgCarry', 'ProgPass', 'Shot', 'OnTarget%', 'Shot conversion', 
                    'Shot/Goal', 'SOG', 'Success1v1', 'Take on into the Box', 
                    'TakeOn', 'ThrghBalls', 'TouchOpBox', 'Touches', 'xA', 
                    'xA +/-', 'xG +/-', 'xGOT'
                ],
                "Defensive Metrics": [
                    'AdjInt', 'AdjTckl', 'Blocks', 'Clrnce', 'Int', 
                    'TcklAtt', 'Tckl', 'TcklMade%', 'TcklA3'
                ],
                "Physical Offensive Metrics": [
                    'PSV-99', 'Distance', 'M/min', 'HSR Distance', 'HSR Count', 
                    'Sprint Distance', 'Sprint Count', 'HI Distance', 
                    'HI Count', 'Medium Acceleration Count', 
                    'High Acceleration Count', 'Medium Deceleration Count', 
                    'High Deceleration Count'
                ],
                "Physical Defensive Metrics": [
                    'Distance OTIP', 'M/min OTIP', 'HSR Distance OTIP', 
                    'HSR Count OTIP', 'Sprint Distance OTIP', 'Sprint Count OTIP',
                    'HI Distance OTIP', 'HI Count OTIP', 'Medium Acceleration Count OTIP', 
                    'High Acceleration Count OTIP', 'Medium Deceleration Count OTIP', 
                    'High Deceleration Count OTIP'
                ]
            }

            # Iterate over each section
            for section, metrics in sections.items():
                st.markdown(f"<h3 style='font-size:15px; color:#333; font-weight:bold;'>{section}</h3>", unsafe_allow_html=True)
                # Iterate over the metrics for the current section
                for metric in metrics:
                    # Display the metric and its explanation in italic
                    explanation = glossary.get(metric, "")
                    st.markdown(f"{metric}: *{explanation}*")
