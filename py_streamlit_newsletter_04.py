import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import gdown

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

# Function to apply custom CSS for mobile responsiveness
def set_mobile_css():
    st.markdown(
        """
        <style>
        /* Your CSS styles */
        /* Example CSS */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Position above the text */
            left: 50%;
            margin-left: -100px;
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
@st.cache_data
def download_and_load_data(file_url, data_version):
    # Define the file path for the downloaded parquet file
    parquet_file = f'/tmp/newupclean7_{data_version}.parquet'

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
    file_url = 'https://drive.google.com/uc?id=1SMirzdQ0cItiGRQ2bccqKpjLw6EIVQWu'
    data_version = 'v2'  # Update this to a new value when your data changes
    data = download_and_load_data(file_url, data_version)

    # Check if the data was loaded successfully
    if data is None:
        st.error("Failed to load data")
        st.stop()
    else:
        # Proceed with your app
        set_mobile_css()
        st.write("Data successfully loaded!")

        # Uncomment the following line to print the columns in your data for debugging
        # st.write("Data columns:", data.columns.tolist())

        # **Initialize necessary variables and minimal processing for filters**

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

        # Adjust column names to match your data
        # Uncomment the following line to print the columns in your data
        # st.write("Data columns:", data.columns.tolist())

        # Based on the data columns, set the correct position column name
        # For example, if 'Position_x' exists, use it; else, check for 'Position'
        if 'Position_x' in data.columns:
            position_column = 'Position_x'
        elif 'Position' in data.columns:
            position_column = 'Position'
        else:
            st.error("Position column not found in the data.")
            st.stop()

        # Assign positions to multiple groups
        data['Position Groups'] = data[position_column].apply(
            lambda pos: [group for group, positions in position_groups.items() if pos in positions])

        # Initialize session state for 'run_clicked'
        if 'run_clicked' not in st.session_state:
            st.session_state['run_clicked'] = False

        def reset_run():
            st.session_state['run_clicked'] = False

        def run_callback():
            st.session_state['run_clicked'] = True

        # Display the logo at the top
        st.image('logo.png', use_column_width=False, width=800)

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

                week_summary['Matchday'] = week_summary.apply(
                    lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})", axis=1
                )

                filtered_weeks = week_summary[week_summary['League'] == selected_league].sort_values(by='min').drop_duplicates(subset=['Week'])

                matchday_options = filtered_weeks['Matchday'].tolist()

                # Replace selectbox with multiselect
                selected_matchdays = st.multiselect("Select Matchdays", matchday_options, key="select_matchdays", on_change=reset_run)

                # If no matchdays are selected, show a warning and stop
                if not selected_matchdays:
                    st.warning("Please select at least one matchday.")
                    st.stop()

                selected_weeks = filtered_weeks[filtered_weeks['Matchday'].isin(selected_matchdays)]['Week'].unique().tolist()

                # Get the last date among the selected matchdays
                selected_dates = filtered_weeks[filtered_weeks['Matchday'].isin(selected_matchdays)]['max']
                last_selected_date = max(selected_dates)

            with col3:
                position_group_options = list(position_groups.keys())
                selected_position_group = st.selectbox("Select Position Group", position_group_options, key="select_position_group", on_change=reset_run)

        # Add the "Run" button
        st.button("Run", on_click=run_callback)

        # Process data only if "Run" has been clicked
        if st.session_state['run_clicked']:
            # **Perform data processing here after "Run" is clicked**

            # Glossary content with metrics integrated
            glossary = {
                'Rating Metrics': '',
                'Overall Rating': 'Player\'s overall performance across all metrics.',
                'Defensive Rating': 'Player\'s overall defensive performance. Metrics: TcklMade%, TcklAtt, Tckl, AdjTckl, TcklA3, Blocks, Int, AdjInt, Clrnce',
                'Goal Threat Rating': 'Player\'s threat to score goals. Metrics: Goal, Shot/Goal, MinPerGoal, ExpG, xGOT, xG +/- , Shot, SOG, Shot conversion, OnTarget%',
                'Offensive Rating': 'Player\'s overall offensive performance. Metrics: 2ndAst, Ast, ExpG, ExpGExPn, Goal, GoalExPn, KeyPass, MinPerChnc, MinPerGoal, PsAtt, PsCmp, Pass%, PsIntoA3rd, PsRec, ProgCarry, ProgPass, Shot, Shot conversion, Shot/Goal, SOG, OnTarget%, Success1v1, Take on into the Box, TakeOn, ThrghBalls, TouchOpBox, Touches, xA, xA +/- , xG +/- , xGOT',
                'Physical Offensive Rating': 'Player\'s physical contributions to offensive play. Metrics: PSV-99, Distance, M/min, HSR Distance, HSR Count, Sprint Distance, Sprint Count, HI Distance, HI Count, Medium Acceleration Count, High Acceleration Count, Medium Deceleration Count, High Deceleration Count',
                'Physical Defensive Rating': 'Player\'s physical contributions to defensive play. Metrics: Distance OTIP, M/min OTIP, HSR Distance OTIP, HSR Count OTIP, Sprint Distance OTIP, Sprint Count OTIP, HI Distance OTIP, HI Count OTIP, Medium Acceleration Count OTIP, High Acceleration Count OTIP, Medium Deceleration Count OTIP, High Deceleration Count OTIP',
                'Offensive Metrics': '',
                '2ndAst': 'The pass that assists the assist leading to a goal.',
                'Ast': 'Assists.',
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
                'xA +/-': 'Expected assists compared to actual assists.',
                'xG +/-': 'Expected goals compared to actual goals.',
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

            # Convert text-based numbers to numeric, handling percentage metrics
            percentage_metrics = ['TcklMade%', 'Pass%', 'OnTarget%']

            # Remove percentage signs and convert to numeric
            for metric in percentage_metrics:
                if metric in data.columns:
                    data[metric] = pd.to_numeric(data[metric].astype(str).str.replace('%', ''), errors='coerce')

            # Convert other text-based numbers to numeric
            physical_metrics = ['PSV-99', 'Distance', 'M/min', 'HSR Distance', 'HSR Count', 'Sprint Distance',
                                'Sprint Count', 'HI Distance', 'HI Count', 'Medium Acceleration Count',
                                'High Acceleration Count', 'Medium Deceleration Count', 'High Deceleration Count',
                                'Distance OTIP', 'M/min OTIP', 'HSR Distance OTIP', 'HSR Count OTIP',
                                'Sprint Distance OTIP', 'Sprint Count OTIP', 'HI Distance OTIP', 'HI Count OTIP',
                                'Medium Acceleration Count OTIP', 'High Acceleration Count OTIP',
                                'Medium Deceleration Count OTIP', 'High Deceleration Count OTIP']

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

            # Combine all metrics for processing
            all_metrics = list(set(
                physical_metrics + offensive_metrics + defensive_metrics + goal_threat_metrics + percentage_metrics
            ))

            for metric in all_metrics:
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

            # Calculate the ratings
            data['Physical Offensive Rating'] = scaler.fit_transform(
                quantile_transformer.fit_transform(data[physical_offensive_metrics].fillna(0))
            ).mean(axis=1)

            data['Physical Defensive Rating'] = scaler.fit_transform(
                quantile_transformer.fit_transform(data[physical_defensive_metrics].fillna(0))
            ).mean(axis=1)

            data['Offensive Rating'] = scaler.fit_transform(
                quantile_transformer.fit_transform(data[offensive_metrics].fillna(0))
            ).mean(axis=1)

            data['Defensive Rating'] = scaler.fit_transform(
                quantile_transformer.fit_transform(data[defensive_metrics].fillna(0))
            ).mean(axis=1)

            data['Goal Threat Rating'] = scaler.fit_transform(
                quantile_transformer.fit_transform(data[goal_threat_metrics].fillna(0))
            ).mean(axis=1)

            # **Add the Overall Rating by combining all metrics**
            # Create a list of all metrics used in the ratings
            rating_metrics = ['Physical Offensive Rating', 'Physical Defensive Rating',
                              'Offensive Rating', 'Defensive Rating', 'Goal Threat Rating']

            data['Overall Rating'] = data[rating_metrics].mean(axis=1)

            # **Calculate Cumulative Averages for Metrics**

            # Ensure the data is sorted
            data = data.sort_values(['League', 'playerFullName', 'Date'])

            # Create a list of metrics for which we want cumulative averages
            metrics_for_cum_avg = rating_metrics + physical_offensive_metrics + physical_defensive_metrics + offensive_metrics + defensive_metrics

            # Remove duplicates
            metrics_for_cum_avg = list(set(metrics_for_cum_avg))

            # Calculate cumulative averages for each player in each league
            for metric in metrics_for_cum_avg:
                data[f'{metric}_cum_avg'] = data.groupby(['League', 'playerFullName'])[metric].expanding().mean().reset_index(level=[0, 1], drop=True)

            # **Updated Filtering:**
            # Filter the data by the selected position group and the selected matchdays
            league_and_position_data = data[
                (data['League'] == selected_league) &
                (data['Week'].isin(selected_weeks)) &
                (data['Position Groups'].apply(lambda groups: selected_position_group in groups))
            ]

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

            # Use a container to make the expandable sections span the full width
            with st.container():
                tooltip_headers = {metric: glossary.get(metric, '') for metric in rating_metrics + physical_metrics + offensive_metrics + defensive_metrics}

                def display_metric_tables(metrics_list, title):
                    with st.expander(title, expanded=False):  # Setting expanded=False to keep it closed by default
                        for metric in metrics_list:
                            if metric not in data.columns:
                                st.write(f"Metric {metric} not found in the data")
                                continue

                            metric_data = league_and_position_data

                            # Determine aggregation function
                            if metric in count_metrics:
                                agg_func = 'sum'
                            elif metric in average_metrics or metric in percentage_metrics:
                                agg_func = 'mean'
                            else:
                                agg_func = 'mean'  # Default to mean if unsure

                            # Define the aggregation dictionary
                            agg_dict = {'Age': 'last', metric: agg_func, f'{metric}_cum_avg': 'last'}

                            # Include 'Team' and 'Position' if they exist
                            if 'Team' in metric_data.columns:
                                agg_dict['Team'] = 'last'
                            if position_column in metric_data.columns:
                                agg_dict[position_column] = 'last'

                            # Perform the aggregation
                            try:
                                latest_data = metric_data.groupby('playerFullName').agg(agg_dict).reset_index()
                            except KeyError as e:
                                st.error(f"Column not found during aggregation: {e}")
                                continue

                            # Round the Age column to ensure no decimals
                            latest_data['Age'] = latest_data['Age'].round(0).astype(int)

                            # Prepare the data
                            columns_to_select = ['playerFullName', 'Age', 'Team', position_column, metric, f'{metric}_cum_avg']
                            available_columns = [col for col in columns_to_select if col in latest_data.columns]
                            top10 = latest_data[available_columns].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

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
                                top10.rename(columns={'playerFullName': 'Player', position_column: 'Position'}, inplace=True)

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

                            # If the metric is 'PSV-99', also display the overall top 10
                            if metric == 'PSV-99':
                                # For 'PSV-99', use data filtered only by league and selected weeks (matchdays), ignore position group
                                metric_data_overall = data[
                                    (data['League'] == selected_league) &
                                    (data['Week'].isin(selected_weeks))
                                ]

                                # Aggregate the data over the selected matchdays
                                agg_dict_overall = {'Age': 'last', metric: 'mean', f'{metric}_cum_avg': 'last'}
                                if 'Team' in metric_data_overall.columns:
                                    agg_dict_overall['Team'] = 'last'
                                if position_column in metric_data_overall.columns:
                                    agg_dict_overall[position_column] = 'last'

                                latest_data_overall = metric_data_overall.groupby('playerFullName').agg(agg_dict_overall).reset_index()

                                # Round the Age column to ensure no decimals
                                latest_data_overall['Age'] = latest_data_overall['Age'].round(0).astype(int)

                                # Prepare the data
                                columns_to_select_overall = ['playerFullName', 'Age', 'Team', position_column, metric, f'{metric}_cum_avg']
                                available_columns_overall = [col for col in columns_to_select_overall if col in latest_data_overall.columns]
                                top10_overall = latest_data_overall[available_columns_overall].dropna(subset=[metric]).sort_values(by=metric, ascending=False).head(10)

                                if top10_overall.empty:
                                    st.header(f"Top 10 Players in {metric} (Overall)")
                                    st.write("No data available")
                                else:
                                    # Reset the index to create a rank column starting from 1
                                    top10_overall.reset_index(drop=True, inplace=True)
                                    top10_overall.index += 1
                                    top10_overall.index.name = 'Rank'

                                    # Ensure the Rank column is part of the DataFrame before styling
                                    top10_overall = top10_overall.reset_index()

                                    st.markdown(f"<h2>{metric} (Overall)</h2>", unsafe_allow_html=True)
                                    top10_overall.rename(columns={'playerFullName': 'Player', position_column: 'Position'}, inplace=True)

                                    # Format the metric value with cumulative average
                                    top10_overall[metric] = top10_overall.apply(
                                        lambda row: f"{row[metric]:.2f} ({row[f'{metric}_cum_avg']:.2f})" if pd.notnull(row[f'{metric}_cum_avg']) else f"{row[metric]:.2f}",
                                        axis=1
                                    )

                                    # Remove the cumulative average column from the DataFrame as it's now included in the metric column
                                    top10_overall.drop(columns=[f'{metric}_cum_avg'], inplace=True)

                                    def color_row(row):
                                        return ['background-color: #d4edda' if row['Age'] < 24 else '' for _ in row]

                                    top10_overall_styled = top10_overall.style.apply(color_row, axis=1)
                                    top10_overall_html = top10_overall_styled.to_html()

                                    for header, tooltip in tooltip_headers.items():
                                        if tooltip:
                                            top10_overall_html = top10_overall_html.replace(f'>{header}<', f'><span class="tooltip">{header}<span class="tooltiptext">{tooltip}</span></span><')

                                    st.write(top10_overall_html, unsafe_allow_html=True)

                # Call the display_metric_tables function with updated metric names
                display_metric_tables(['Overall Rating', 'Offensive Rating', 'Goal Threat Rating', 'Defensive Rating', 'Physical Offensive Rating', 'Physical Defensive Rating'], "Rating Metrics")
                display_metric_tables(physical_offensive_metrics, "Physical Offensive Metrics")
                display_metric_tables(physical_defensive_metrics, "Physical Defensive Metrics")
                display_metric_tables(offensive_metrics, "Offensive Metrics")
                display_metric_tables(defensive_metrics, "Defensive Metrics")

            # Glossary section - Render only after authentication inside an expander
            with st.expander("Glossary"):
                sections = {
                    "Rating Metrics": [
                        'Overall Rating', 'Defensive Rating', 'Goal Threat Rating', 'Offensive Rating',
                        'Physical Defensive Rating', 'Physical Offensive Rating'
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
                    "Physical Defensive Metrics": physical_defensive_metrics
                }

                # Iterate over each section
                for section, metrics in sections.items():
                    st.markdown(f"<h3 style='font-size:15px; color:#333; font-weight:bold;'>{section}</h3>", unsafe_allow_html=True)
                    # Iterate over the metrics for the current section
                    for metric in metrics:
                        # Display the metric and its explanation in italic
                        explanation = glossary.get(metric, "")
                        st.markdown(f"{metric}: *{explanation}*")

        else:
            st.write("Please set your filters and click 'Run' to display the data.")
