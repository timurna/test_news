import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import gdown
import numpy as np  # <-- We import NumPy for the log1p function

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

# Initialize data
data = None

# Ensure 'authenticated' is initialized in session state
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

def authenticate(username, password):
    try:
        stored_username = st.secrets["credentials"]["username"]
        stored_password = st.secrets["credentials"]["password"]
    except KeyError as e:
        st.error(f"Error: {e}. Credentials not found in Streamlit secrets.")
        return False
    return username == stored_username and password == stored_password

def login():
    # Store username and password in session state to maintain values across reruns
    if 'login_username' not in st.session_state:
        st.session_state.login_username = ''
    if 'login_password' not in st.session_state:
        st.session_state.login_password = ''

    st.text_input("Username", key="login_username")
    st.text_input("Password", type="password", key="login_password")

    def authenticate_and_login():
        username = st.session_state.login_username
        password = st.session_state.login_password
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

    st.button("Login", on_click=authenticate_and_login)

# Function to apply custom CSS for tooltips
def set_mobile_css():
    st.markdown(
        """
        <style>
        /* Tooltip CSS */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 250px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 10px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the text */
            left: 50%;
            margin-left: -125px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
            line-height: 1.2;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to download and load the file from Google Drive
@st.cache_data
def download_and_load_data(file_url, data_version):
    # Define the file path for the downloaded parquet file
    parquet_file = f'/tmp/newup5_{data_version}.parquet'

    # Download the file using gdown with fuzzy=True
    try:
        gdown.download(url=file_url, output=parquet_file, quiet=False, fuzzy=True)
    except Exception as e:
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

# Ensure proper authentication
if not st.session_state.authenticated:
    login()
else:
    # User is authenticated
    st.write("Welcome! You are logged in.")

    # Load the dataset **only** after successful login
    file_url = 'https://drive.google.com/uc?id=1Z6jrYJW0qtALOsLhARvnP0voGIDuRIQA'
    data_version = 'v1'  # Update this to a new value when your data changes
    data = download_and_load_data(file_url, data_version)

    # Check if the data was loaded successfully
    if data is None:
        st.error("Failed to load data")
        st.stop()
    else:
        # Proceed with your app
        set_mobile_css()
        st.write("Data successfully loaded!")

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

        # Based on the data columns, set the correct position column name
        if 'Position_x' in data.columns:
            position_column = 'Position_x'
        elif 'Position' in data.columns:
            position_column = 'Position'
        else:
            st.error("Position column not found in the data.")
            st.stop()

        # Assign positions to multiple groups
        data['Position Groups'] = data[position_column].apply(
            lambda pos: [group for group, positions in position_groups.items() if pos in positions]
        )

        # Initialize session state for 'run_clicked'
        if 'run_clicked' not in st.session_state:
            st.session_state['run_clicked'] = False

        def reset_run():
            st.session_state['run_clicked'] = False

        def run_callback():
            st.session_state['run_clicked'] = True

        # Display the logo at the top
        st.image('logo.png', use_container_width=True, width=800)

        # Create a single row for all the filters
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                leagues = sorted(data['League'].unique())  # Sort leagues alphabetically
                selected_league = st.selectbox("Select League", leagues, key="select_league", on_change=reset_run)

            with col2:
                league_data = data[data['League'] == selected_league]

                # Week Summary and Matchday Filtering Logic
                week_summary = league_data.groupby(['League', 'Week']).agg({'Date': ['min', 'max']}).reset_index()
                week_summary.columns = ['League', 'Week', 'min', 'max']

                week_summary['min'] = pd.to_datetime(week_summary['min'])
                week_summary['max'] = pd.to_datetime(week_summary['max'])

                # Convert Week to int to avoid decimals in Matchday display
                week_summary['Week'] = week_summary['Week'].astype(int)

                week_summary['Matchday'] = week_summary.apply(
                    lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})",
                    axis=1
                )

                filtered_weeks = (
                    week_summary[week_summary['League'] == selected_league]
                    .sort_values(by='max', ascending=False)
                    .drop_duplicates(subset=['Week'])
                )

                # Calculate the number of matchdays
                num_matchdays = len(filtered_weeks)

                # Create 'All (X)' option
                all_option = f"All ({num_matchdays})"

                # Add 'All (X)' at the top of matchday options
                matchday_options = [all_option] + filtered_weeks['Matchday'].tolist()

                # Replace selectbox with multiselect including 'All'
                selected_matchdays = st.multiselect(
                    "Select Matchdays",
                    matchday_options,
                    key="select_matchdays",
                    on_change=reset_run
                )

                # If no matchdays are selected, show a warning and stop
                if not selected_matchdays:
                    st.warning("Please select at least one matchday.")
                    st.stop()

                # If 'All (X)' is selected, use all matchdays
                if all_option in selected_matchdays:
                    selected_matchdays = filtered_weeks['Matchday'].tolist()

                selected_weeks = filtered_weeks[
                    filtered_weeks['Matchday'].isin(selected_matchdays)
                ]['Week'].unique().tolist()

                # Get the last date among the selected matchdays
                selected_dates = filtered_weeks[
                    filtered_weeks['Matchday'].isin(selected_matchdays)
                ]['max']
                last_selected_date = max(selected_dates)

            with col3:
                position_group_options = list(position_groups.keys())
                selected_position_group = st.selectbox(
                    "Select Position Group",
                    position_group_options,
                    key="select_position_group",
                    on_change=reset_run
                )

        # Add the "Run" button
        st.button("Run", on_click=run_callback)

        # Process data only if "Run" has been clicked
        if st.session_state['run_clicked']:
            # Glossary content with metrics integrated
            glossary = {
                'Ratings': 'A collection of aggregated performance metrics.',
                'Overall Rating': 'Player\'s overall performance across all metrics.',
                'Defensive Rating': 'Player\'s overall defensive performance. Metrics include various defensive actions.',
                'Goal Threat Rating': 'Player\'s threat to score goals. Metrics include shots, expected goals, etc.',
                'Offensive Rating': 'Player\'s overall offensive performance. Metrics include assists, key passes, etc.',
                'Physical Offensive Rating': 'Player\'s physical contributions to offensive play.',
                'Physical Defensive Rating': 'Player\'s physical contributions to defensive play.',
                'Pass Rating': 'Player\'s overall passing performance. Metrics include pass attempts, completion, key passes, through balls, etc.',
                'Activity Rating': 'Player\'s involvement in moves, receiving passes, and touches in advanced areas. (Touches, TouchOpBox, PsRec)',
                'Ballcarrier Rating': 'Player\'s ability to carry the ball forward and beat opponents. (TakeOn, Success1v1, Take on into the Box, ProgCarry)',
                'Min': 'Minutes played in the selected matchday(s) (total minutes played across all matchdays)',
                # Physical Metrics
                'PSV-99': 'Player\'s physical performance score compared to peers.',
                'Distance': 'Total distance covered by the player.',
                'M/min': 'Meters covered per minute.',
                'HSR Distance': 'High-speed running distance.',
                'HSR Count': 'Number of high-speed runs.',
                'Sprint Distance': 'Total sprint distance.',
                'Sprint Count': 'Number of sprints.',
                'HI Distance': 'High-intensity running distance.',
                'HI Count': 'Number of high-intensity runs.',
                'Medium Acceleration Count': 'Number of medium accelerations.',
                'High Acceleration Count': 'Number of high accelerations.',
                'Medium Deceleration Count': 'Number of medium decelerations.',
                'High Deceleration Count': 'Number of high decelerations.',
                'Distance OTIP': 'Distance covered out of team possession.',
                'M/min OTIP': 'Meters per minute out of team possession.',
                'HSR Distance OTIP': 'High-speed running distance out of team possession.',
                'HSR Count OTIP': 'High-speed run count out of team possession.',
                'Sprint Distance OTIP': 'Sprint distance out of team possession.',
                'Sprint Count OTIP': 'Sprint count out of team possession.',
                'HI Distance OTIP': 'High-intensity distance out of team possession.',
                'HI Count OTIP': 'High-intensity count out of team possession.',
                'Medium Acceleration Count OTIP': 'Medium accelerations out of team possession.',
                'High Acceleration Count OTIP': 'High accelerations out of team possession.',
                'Medium Deceleration Count OTIP': 'Medium decelerations out of team possession.',
                'High Deceleration Count OTIP': 'High decelerations out of team possession.',
                # Offensive Metrics
                '2ndAst': 'Secondary assists.',
                'Ast': 'Assists.',
                'ExpG': 'Expected goals.',
                'ExpGExPn': 'Expected goals excluding penalties.',
                'Goal': 'Goals scored.',
                'GoalExPn': 'Goals excluding penalties.',
                'KeyPass': 'Passes leading directly to a shot.',
                'MinPerChnc': 'Minutes per chance created.',
                'MinPerGoal': 'Minutes per goal scored.',
                'PsAtt': 'Passes attempted.',
                'PsCmp': 'Passes completed.',
                'Pass%': 'Pass completion percentage.',
                'PsIntoA3rd': 'Passes into the attacking third.',
                'PsRec': 'Passes received.',
                'ProgCarry': 'Progressive carries.',
                'ProgPass': 'Progressive passes.',
                'Shot': 'Shots taken.',
                'Shot conversion': 'Percentage of shots resulting in goals.',
                'Shot/Goal': 'Shots per goal.',
                'SOG': 'Shots on goal.',
                'OnTarget%': 'Percentage of shots on target.',
                'Success1v1': 'Successful one-on-one take-ons.',
                'Take on into the Box': 'Take-ons into the penalty box.',
                'TakeOn': 'Total take-ons attempted.',
                'ThrghBalls': 'Through balls.',
                'TouchOpBox': 'Touches in the opposition box.',
                'Touches': 'Total touches.',
                'xA': 'Expected assists.',
                'xA +/-': 'Expected assists above or below average.',
                'xG +/-': 'Expected goals above or below average.',
                'xGOT': 'Expected goals on target.',
                # Defensive Metrics
                'AdjInt': 'Adjusted interceptions.',
                'AdjTckl': 'Adjusted tackles.',
                'Blocks': 'Blocks made.',
                'Clrnce': 'Clearances.',
                'Int': 'Interceptions.',
                'TcklAtt': 'Tackle attempts.',
                'Tckl': 'Tackles made.',
                'TcklMade%': 'Percentage of successful tackles.',
                'TcklA3': 'Tackles in the attacking third.',
                # Percentage Metrics
                'OnTarget%': 'Percentage of shots on target.',
                'Pass%': 'Pass completion percentage.',
                'TcklMade%': 'Percentage of tackles made successfully.',
            }

            # Perform data processing steps here
            # Calculate age from birthdate
            data['DOB'] = pd.to_datetime(data['DOB'])
            today = datetime.today()
            data['Age'] = data['DOB'].apply(
                lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day))
            )

            # Ensure 'Date' is in datetime format
            data['Date'] = pd.to_datetime(data['Date'])

            # Convert text-based numbers to numeric, handling percentage metrics
            percentage_metrics = ['TcklMade%', 'Pass%', 'OnTarget%']

            # Remove percentage signs and convert to numeric
            for metric in percentage_metrics:
                if metric in data.columns:
                    data[metric] = pd.to_numeric(
                        data[metric].astype(str).str.replace('%', ''),
                        errors='coerce'
                    )

            # Define the main metric groups
            physical_metrics = [
                'PSV-99', 'Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance',
                'Sprint Count', 'HI Distance', 'HI Count', 'Medium Acceleration Count',
                'High Acceleration Count', 'Medium Deceleration Count', 'High Deceleration Count',
                'Distance OTIP', 'M/min OTIP', 'HSR Distance OTIP', 'HSR Count OTIP',
                'Sprint Distance OTIP', 'Sprint Count OTIP', 'HI Distance OTIP', 'HI Count OTIP',
                'Medium Acceleration Count OTIP', 'High Acceleration Count OTIP',
                'Medium Deceleration Count OTIP', 'High Deceleration Count OTIP'
            ]

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

            # NEW: Activity & Ballcarrier metric groups
            activity_metrics = ['Touches', 'TouchOpBox', 'PsRec']
            ballcarrier_metrics = ['TakeOn', 'Success1v1', 'Take on into the Box', 'ProgCarry']

            # Combine all metrics for processing
            all_metrics = list(set(
                physical_metrics
                + offensive_metrics
                + defensive_metrics
                + goal_threat_metrics
                + percentage_metrics
                + activity_metrics
                + ballcarrier_metrics
            ))

            # Convert numeric columns (excluding the already handled percentages)
            for metric in all_metrics:
                if metric in data.columns and metric not in percentage_metrics:
                    data[metric] = pd.to_numeric(
                        data[metric].astype(str).str.replace(',', '.'),
                        errors='coerce'
                    )

            # Ensure 'Min' column is numeric
            if 'Min' in data.columns:
                data['Min'] = pd.to_numeric(data['Min'], errors='coerce')
            else:
                st.error("Column 'Min' not found in data.")
                st.stop()

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
            fill_na_conditionally(data, activity_metrics)
            fill_na_conditionally(data, ballcarrier_metrics)

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

            # Define pass metrics (with weighting logic already in place before log transform)
            pass_metrics = [
                'PsAtt', 'PsCmp', 'Pass%', 'PsIntoA3rd', 'KeyPass', 'ThrghBalls'
            ]
            weights = {
                'PsAtt': 1.0,
                'PsCmp': 1.0,
                'Pass%': 1.0,
                'PsIntoA3rd': 2.0,
                'KeyPass': 2.0,
                'ThrghBalls': 2.0
            }

            # --- Calculate the Ratings ---

            # 1) Physical Offensive Rating
            data['Physical Offensive Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[physical_offensive_metrics].fillna(0))
                ).mean(axis=1)
            )

            # 2) Physical Defensive Rating
            data['Physical Defensive Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[physical_defensive_metrics].fillna(0))
                ).mean(axis=1)
            )

            # 3) Offensive Rating
            data['Offensive Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[offensive_metrics].fillna(0))
                ).mean(axis=1)
            )

            # 4) Defensive Rating
            data['Defensive Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[defensive_metrics].fillna(0))
                ).mean(axis=1)
            )

            # 5) Goal Threat Rating
            data['Goal Threat Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[goal_threat_metrics].fillna(0))
                ).mean(axis=1)
            )

            # 6) Pass Rating with LOG TRANSFORM + weighting **before** log
            pass_subset = data[pass_metrics].fillna(0).copy()
            for col in pass_metrics:
                pass_subset[col] = pass_subset[col] * weights[col]

            pass_logged = np.log1p(pass_subset)
            pass_transformed = quantile_transformer.fit_transform(pass_logged)
            pass_scaled = scaler.fit_transform(pass_transformed)
            data['Pass Rating'] = pass_scaled.mean(axis=1)

            # 7) Activity Rating (no log, no extra weighting)
            data['Activity Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[activity_metrics].fillna(0))
                ).mean(axis=1)
            )

            # 8) Ballcarrier Rating (no log, no extra weighting)
            data['Ballcarrier Rating'] = (
                scaler.fit_transform(
                    quantile_transformer.fit_transform(data[ballcarrier_metrics].fillna(0))
                ).mean(axis=1)
            )

            # Gather all rating columns
            rating_metrics = [
                'Overall Rating',
                'Physical Offensive Rating',
                'Physical Defensive Rating',
                'Offensive Rating',
                'Defensive Rating',
                'Goal Threat Rating',
                'Pass Rating',
                'Activity Rating',
                'Ballcarrier Rating'
            ]

            # Calculate Overall Rating (including Pass Rating). 
            # NOTE: If you want to include the new ratings in Overall Rating, add them here.
            data['Overall Rating'] = data[[
                'Physical Offensive Rating',
                'Physical Defensive Rating',
                'Offensive Rating',
                'Defensive Rating',
                'Goal Threat Rating',
                'Pass Rating'
            ]].mean(axis=1)

            # Sort data
            data = data.sort_values(['League', 'playerFullName', 'Date'])

            # Create a list of metrics for which we want cumulative averages
            metrics_for_cum_avg = (
                rating_metrics
                + physical_offensive_metrics
                + physical_defensive_metrics
                + offensive_metrics
                + defensive_metrics
                + activity_metrics
                + ballcarrier_metrics
            )
            metrics_for_cum_avg = list(set(metrics_for_cum_avg))

            # Calculate cumulative averages for each player in each league
            for metric in metrics_for_cum_avg:
                data[f'{metric}_cum_avg'] = (
                    data
                    .groupby(['League', 'playerFullName'])[metric]
                    .expanding()
                    .mean()
                    .reset_index(level=[0, 1], drop=True)
                )

            # Filter data by the selected position group and the selected matchdays
            league_and_position_data = data[
                (data['League'] == selected_league)
                & (data['Week'].isin(selected_weeks))
                & (data['Position Groups'].apply(lambda groups: selected_position_group in groups))
            ]

            # Data filtered by League and Position Group only (all matchdays)
            league_position_all_data = data[
                (data['League'] == selected_league)
                & (data['Position Groups'].apply(lambda groups: selected_position_group in groups))
            ]

            # Identify the team column globally
            if 'Team' in data.columns:
                team_column = 'Team'
            elif 'Team_x' in data.columns:
                team_column = 'Team_x'
            elif 'Squad' in data.columns:
                team_column = 'Squad'
            else:
                st.warning("Team column not found in data.")
                team_column = None

            # Define metrics that are counts and should be summed
            count_metrics = [
                'Goal', 'Ast', 'KeyPass', 'Shot', 'SOG', 'TakeOn', 'Success1v1', 'Blocks', 'Int', 'Clrnce',
                'Tckl', 'AdjTckl', 'TcklAtt', 'AdjInt', 'TcklA3', 'ThrghBalls', 'TouchOpBox', 'Touches',
                'Take on into the Box', '2ndAst', 'PsAtt', 'PsCmp', 'PsIntoA3rd', 'PsRec', 'ProgCarry', 'ProgPass',
                'Shot conversion', 'Shot/Goal', 'HI Count', 'HI Count OTIP', 'Medium Acceleration Count',
                'Medium Acceleration Count OTIP', 'Medium Deceleration Count', 'Medium Deceleration Count OTIP',
                'High Acceleration Count', 'High Acceleration Count OTIP', 'High Deceleration Count',
                'High Deceleration Count OTIP', 'HSR Count', 'HSR Count OTIP', 'Sprint Count', 'Sprint Count OTIP'
            ]

            # Define metrics that should be averaged
            average_metrics = [
                'Pass%', 'OnTarget%', 'TcklMade%', 'ExpG', 'ExpGExPn', 'xA', 'xG +/-', 'xA +/-', 'xGOT',
                'MinPerGoal', 'MinPerChnc', 'PSV-99', 'Distance', 'Distance OTIP', 'M/min', 'M/min OTIP', 'HI Distance',
                'HI Distance OTIP', 'HSR Distance', 'HSR Distance OTIP', 'Sprint Distance', 'Sprint Distance OTIP'
            ] + rating_metrics

            # Collect Mentions Over All Matchdays
            all_weeks = league_position_all_data['Week'].unique()
            mentions_dict = {}
            rating_metrics_to_collect = [
                'Overall Rating',
                'Offensive Rating',
                'Goal Threat Rating',
                'Pass Rating',
                'Activity Rating',
                'Ballcarrier Rating',
                'Defensive Rating',
                'Physical Offensive Rating',
                'Physical Defensive Rating'
            ]

            for week in all_weeks:
                week_data = league_position_all_data[league_position_all_data['Week'] == week]

                for metric in rating_metrics_to_collect:
                    if metric not in data.columns:
                        continue

                    if metric in count_metrics:
                        agg_func = 'sum'
                    elif metric in average_metrics or metric in percentage_metrics:
                        agg_func = 'mean'
                    else:
                        agg_func = 'mean'

                    agg_dict = {'Age': 'last', metric: agg_func, 'Min': 'sum'}

                    if team_column:
                        agg_dict[team_column] = 'last'

                    if position_column in week_data.columns:
                        agg_dict[position_column] = 'last'

                    try:
                        aggregated_data = week_data.groupby('playerFullName').agg(agg_dict).reset_index()
                    except KeyError as e:
                        st.error(f"Column not found during aggregation for mentions: {e}")
                        continue

                    top10 = aggregated_data.dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

                    for _, row in top10.iterrows():
                        player = row['playerFullName']
                        if player not in mentions_dict:
                            mentions_dict[player] = {
                                'Player': player,
                                'Age': row['Age'],
                                'Team': row.get(team_column, ''),
                                'Position': row.get(position_column, ''),
                                'Total Mentions': 0
                            }
                            for m in rating_metrics_to_collect:
                                mentions_dict[player][m] = 0
                        mentions_dict[player]['Total Mentions'] += 1
                        mentions_dict[player][metric] += 1

            with st.container():
                tooltip_headers = {
                    metric: glossary.get(metric, '')
                    for metric in rating_metrics
                    + physical_metrics
                    + offensive_metrics
                    + defensive_metrics
                    + activity_metrics
                    + ballcarrier_metrics
                }

                # Ratings Section
                with st.expander("Ratings", expanded=False):
                    for metric in rating_metrics_to_collect:
                        if metric not in data.columns:
                            continue

                        metric_data = league_and_position_data

                        if metric in count_metrics:
                            agg_func = 'sum'
                        elif metric in average_metrics or metric in percentage_metrics:
                            agg_func = 'mean'
                        else:
                            agg_func = 'mean'

                        agg_dict = {
                            'Age': 'last',
                            metric: agg_func,
                            f'{metric}_cum_avg': 'last',
                            'Min': 'sum'
                        }

                        if team_column:
                            agg_dict[team_column] = 'last'

                        if position_column in metric_data.columns:
                            agg_dict[position_column] = 'last'

                        try:
                            latest_data = metric_data.groupby('playerFullName').agg(agg_dict).reset_index()
                        except KeyError as e:
                            st.error(f"Column not found during aggregation: {e}")
                            continue

                        latest_data['Age'] = latest_data['Age'].round(0).astype(int)

                        minutes_total = data.groupby('playerFullName')['Min'].sum().reset_index()
                        minutes_total.rename(columns={'Min': 'Min_Total'}, inplace=True)

                        latest_data = latest_data.merge(minutes_total, on='playerFullName', how='left')
                        latest_data['Min'] = latest_data.apply(
                            lambda row: f"{int(row['Min'])} ({int(row['Min_Total'])})",
                            axis=1
                        )

                        columns_to_select = [
                            'playerFullName',
                            'Age',
                            team_column,
                            position_column,
                            'Min',
                            metric,
                            f'{metric}_cum_avg'
                        ]
                        available_columns = [col for col in columns_to_select if col in latest_data.columns]
                        top10 = (
                            latest_data[available_columns]
                            .dropna(subset=[metric])
                            .sort_values(by=metric, ascending=False)
                            .head(10)
                        )

                        if top10.empty:
                            st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)
                            st.write("No data available")
                            continue

                        top10.reset_index(drop=True, inplace=True)
                        top10['Rank'] = top10.index + 1
                        cols = ['Rank'] + [col for col in top10.columns if col != 'Rank']
                        top10 = top10[cols]
                        top10.set_index('Rank', inplace=True)
                        top10.rename(columns={'playerFullName': 'Player', position_column: 'Position'}, inplace=True)

                        if team_column:
                            top10.rename(columns={team_column: 'Team'}, inplace=True)

                        top10[metric] = top10.apply(
                            lambda row: f"{row[metric]:.2f} ({row[f'{metric}_cum_avg']:.2f})"
                            if pd.notnull(row[f'{metric}_cum_avg']) else f"{row[metric]:.2f}",
                            axis=1
                        )
                        top10.drop(columns=[f'{metric}_cum_avg'], inplace=True)
                        cols = ['Player', 'Age', 'Team', 'Position', 'Min', metric]
                        top10 = top10[cols]

                        st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)

                        def color_row(row):
                            if row['Age'] < 24:
                                return ['background-color: #d4edda'] * len(row)
                            else:
                                return [''] * len(row)

                        top10_styled = top10.style.apply(color_row, axis=1)
                        st.dataframe(top10_styled)

                    if mentions_dict:
                        mentions_df = pd.DataFrame.from_dict(mentions_dict, orient='index')
                        cols = ['Player', 'Age', 'Team', 'Position', 'Total Mentions'] + rating_metrics_to_collect
                        mentions_df = mentions_df[cols]
                        mentions_df['Age'] = mentions_df['Age'].round(0).astype(int)
                        mentions_df = mentions_df.sort_values(by='Total Mentions', ascending=False)
                        mentions_df.reset_index(drop=True, inplace=True)
                        mentions_df['Rank'] = mentions_df.index + 1
                        mentions_df.set_index('Rank', inplace=True)
                        st.markdown("<h2>Most Mentioned Players</h2>", unsafe_allow_html=True)

                        def color_row(row):
                            if row['Age'] < 24:
                                return ['background-color: #d4edda'] * len(row)
                            else:
                                return [''] * len(row)

                        mentions_styled = mentions_df.style.apply(color_row, axis=1)
                        st.dataframe(mentions_styled)

                # Function to display other metric tables (including PSV-99 overall top 10)
                def display_metric_tables(metrics_list, title):
                    with st.expander(title, expanded=False):
                        for metric in metrics_list:
                            if metric not in data.columns:
                                st.write(f"Metric {metric} not found in the data")
                                continue

                            metric_data = league_and_position_data

                            if metric in count_metrics:
                                agg_func = 'sum'
                            elif metric in average_metrics or metric in percentage_metrics:
                                agg_func = 'mean'
                            else:
                                agg_func = 'mean'

                            agg_dict = {
                                'Age': 'last',
                                metric: agg_func,
                                f'{metric}_cum_avg': 'last',
                                'Min': 'sum'
                            }

                            if team_column:
                                agg_dict[team_column] = 'last'

                            if position_column in metric_data.columns:
                                agg_dict[position_column] = 'last'

                            try:
                                latest_data = metric_data.groupby('playerFullName').agg(agg_dict).reset_index()
                            except KeyError as e:
                                st.error(f"Column not found during aggregation: {e}")
                                continue

                            latest_data['Age'] = latest_data['Age'].round(0).astype(int)

                            minutes_total = data.groupby('playerFullName')['Min'].sum().reset_index()
                            minutes_total.rename(columns={'Min': 'Min_Total'}, inplace=True)
                            latest_data = latest_data.merge(minutes_total, on='playerFullName', how='left')
                            latest_data['Min'] = latest_data.apply(
                                lambda row: f"{int(row['Min'])} ({int(row['Min_Total'])})",
                                axis=1
                            )

                            columns_to_select = [
                                'playerFullName', 'Age', team_column, position_column,
                                'Min', metric, f'{metric}_cum_avg'
                            ]
                            available_columns = [col for col in columns_to_select if col in latest_data.columns]
                            top10 = (
                                latest_data[available_columns]
                                .dropna(subset=[metric])
                                .sort_values(by=metric, ascending=False)
                                .head(10)
                            )

                            st.markdown(f"<h2>{metric}</h2>", unsafe_allow_html=True)

                            if top10.empty:
                                st.write("No data available")
                            else:
                                top10.reset_index(drop=True, inplace=True)
                                top10['Rank'] = top10.index + 1
                                cols = ['Rank'] + [col for col in top10.columns if col != 'Rank']
                                top10 = top10[cols]
                                top10.set_index('Rank', inplace=True)
                                top10.rename(columns={'playerFullName': 'Player', position_column: 'Position'}, inplace=True)

                                if team_column:
                                    top10.rename(columns={team_column: 'Team'}, inplace=True)

                                top10[metric] = top10.apply(
                                    lambda row: f"{row[metric]:.2f} ({row[f'{metric}_cum_avg']:.2f})"
                                    if pd.notnull(row[f'{metric}_cum_avg']) else f"{row[metric]:.2f}",
                                    axis=1
                                )
                                top10.drop(columns=[f'{metric}_cum_avg'], inplace=True)
                                cols = ['Player', 'Age', 'Team', 'Position', 'Min', metric]
                                top10 = top10[cols]

                                def color_row(row):
                                    if row['Age'] < 24:
                                        return ['background-color: #d4edda'] * len(row)
                                    else:
                                        return [''] * len(row)

                                top10_styled = top10.style.apply(color_row, axis=1)
                                st.dataframe(top10_styled)

                            # If the metric is 'PSV-99', also display the overall top 10 (ignoring position group)
                            if metric == 'PSV-99':
                                metric_data_overall = data[
                                    (data['League'] == selected_league)
                                    & (data['Week'].isin(selected_weeks))
                                ]

                                agg_dict_overall = {
                                    'Age': 'last',
                                    metric: agg_func,
                                    f'{metric}_cum_avg': 'last',
                                    'Min': 'sum'
                                }
                                if team_column:
                                    agg_dict_overall[team_column] = 'last'
                                if position_column in metric_data_overall.columns:
                                    agg_dict_overall[position_column] = 'last'

                                try:
                                    latest_data_overall = metric_data_overall.groupby('playerFullName').agg(agg_dict_overall).reset_index()
                                except KeyError as e:
                                    st.error(f"Column not found during aggregation: {e}")
                                    continue

                                latest_data_overall['Age'] = latest_data_overall['Age'].round(0).astype(int)
                                minutes_total_overall = data.groupby('playerFullName')['Min'].sum().reset_index()
                                minutes_total_overall.rename(columns={'Min': 'Min_Total'}, inplace=True)
                                latest_data_overall = latest_data_overall.merge(minutes_total_overall, on='playerFullName', how='left')
                                latest_data_overall['Min'] = latest_data_overall.apply(
                                    lambda row: f"{int(row['Min'])} ({int(row['Min_Total'])})",
                                    axis=1
                                )

                                columns_to_select_overall = [
                                    'playerFullName', 'Age', team_column, position_column,
                                    'Min', metric, f'{metric}_cum_avg'
                                ]
                                available_columns_overall = [
                                    col for col in columns_to_select_overall if col in latest_data_overall.columns
                                ]
                                top10_overall = (
                                    latest_data_overall[available_columns_overall]
                                    .dropna(subset=[metric])
                                    .sort_values(by=metric, ascending=False)
                                    .head(10)
                                )

                                if not top10_overall.empty:
                                    top10_overall.reset_index(drop=True, inplace=True)
                                    top10_overall['Rank'] = top10_overall.index + 1
                                    cols_overall = ['Rank'] + [col for col in top10_overall.columns if col != 'Rank']
                                    top10_overall = top10_overall[cols_overall]
                                    top10_overall.set_index('Rank', inplace=True)

                                    top10_overall.rename(columns={'playerFullName': 'Player', position_column: 'Position'}, inplace=True)
                                    if team_column:
                                        top10_overall.rename(columns={team_column: 'Team'}, inplace=True)

                                    top10_overall[metric] = top10_overall.apply(
                                        lambda row: f"{row[metric]:.2f} ({row[f'{metric}_cum_avg']:.2f})"
                                        if pd.notnull(row[f'{metric}_cum_avg']) else f"{row[metric]:.2f}",
                                        axis=1
                                    )

                                    top10_overall.drop(columns=[f'{metric}_cum_avg'], inplace=True)
                                    cols_overall = ['Player', 'Age', 'Team', 'Position', 'Min', metric]
                                    top10_overall = top10_overall[cols_overall]

                                    st.markdown(f"<h2>{metric} (Overall Top 10)</h2>", unsafe_allow_html=True)

                                    def color_row_overall(row):
                                        if row['Age'] < 24:
                                            return ['background-color: #d4edda'] * len(row)
                                        else:
                                            return [''] * len(row)

                                    top10_overall_styled = top10_overall.style.apply(color_row_overall, axis=1)
                                    st.dataframe(top10_overall_styled)

                # Display metric tables
                display_metric_tables(physical_offensive_metrics, "Physical Offensive Metrics")
                display_metric_tables(physical_defensive_metrics, "Physical Defensive Metrics")
                display_metric_tables(offensive_metrics, "Offensive Metrics")
                display_metric_tables(defensive_metrics, "Defensive Metrics")

            # Glossary section
            with st.expander("Glossary"):
                sections = {
                    "Ratings": [
                        'Overall Rating',
                        'Defensive Rating',
                        'Goal Threat Rating',
                        'Offensive Rating',
                        'Physical Defensive Rating',
                        'Physical Offensive Rating',
                        'Pass Rating',
                        'Activity Rating',
                        'Ballcarrier Rating'
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
                    "Physical Offensive Metrics": physical_offensive_metrics,
                    "Physical Defensive Metrics": physical_defensive_metrics,
                    "Activity Metrics": activity_metrics,
                    "Ballcarrier Metrics": ballcarrier_metrics
                }

                for section, metrics in sections.items():
                    st.markdown(
                        f"<h3 style='font-size:15px; color:#333; font-weight:bold;'>{section}</h3>",
                        unsafe_allow_html=True
                    )
                    for metric in metrics:
                        explanation = glossary.get(metric, "")
                        st.markdown(f"{metric}: *{explanation}*")

        else:
            st.write("Please set your filters and click 'Run' to display the data.")
