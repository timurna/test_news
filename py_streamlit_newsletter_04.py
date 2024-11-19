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
    parquet_file = f'/tmp/newupclean8_{data_version}.parquet'

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
    file_url = 'https://drive.google.com/uc?id=1v7yB6MnNMSPqwJyOijqw1aqYVTOx9MPK'
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

                # Sort the weeks by 'max' date in descending order to have the latest matchday on top
                filtered_weeks = week_summary[week_summary['League'] == selected_league].sort_values(by='max', ascending=False).drop_duplicates(subset=['Week'])

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

            # [ ... existing code for data processing and metric calculations ... ]
            # (For brevity, I'm not repeating all the existing code here.)

            # Ensure the data is sorted
            data = data.sort_values(['League', 'playerFullName', 'Date'])

            # [ ... code to calculate cumulative averages ... ]

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
                'MinPerGoal', 'MinPerChnc', 'Distance', 'Distance OTIP', 'M/min', 'M/min OTIP', 'HI Distance',
                'HI Distance OTIP', 'HSR Distance', 'HSR Distance OTIP', 'Sprint Distance', 'Sprint Distance OTIP'
            ] + rating_metrics

            # Define metrics for which we want the maximum value
            max_metrics = ['PSV-99']

            # Remove 'PSV-99' from average_metrics if it's there
            if 'PSV-99' in average_metrics:
                average_metrics.remove('PSV-99')

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
                            elif metric in max_metrics:
                                agg_func = 'max'
                            else:
                                agg_func = 'mean'  # Default to mean if unsure

                            # Identify the team column
                            if 'Team' in metric_data.columns:
                                team_column = 'Team'
                            elif 'Team_x' in metric_data.columns:
                                team_column = 'Team_x'
                            elif 'Squad' in metric_data.columns:
                                team_column = 'Squad'
                            else:
                                st.warning("Team column not found in data.")
                                team_column = None

                            # Prepare the aggregation dictionary using named aggregations
                            agg_dict = {
                                'Age': 'last',
                            }

                            if team_column:
                                agg_dict['Team'] = (team_column, 'last')
                            if position_column in metric_data.columns:
                                agg_dict['Position'] = (position_column, 'last')

                            if metric == 'PSV-99':
                                agg_dict[f'{metric}_max'] = (metric, 'max')
                                agg_dict[f'{metric}_avg_over_selected'] = (metric, 'mean')
                            else:
                                agg_dict[metric] = (metric, agg_func)
                                agg_dict[f'{metric}_cum_avg'] = (f'{metric}_cum_avg', 'last')

                            # Perform the aggregation
                            try:
                                latest_data = metric_data.groupby('playerFullName').agg(**agg_dict).reset_index()
                            except KeyError as e:
                                st.error(f"Column not found during aggregation: {e}")
                                continue

                            # Round the Age column to ensure no decimals
                            latest_data['Age'] = latest_data['Age'].round(0).astype(int)

                            # Prepare the data
                            if metric == 'PSV-99':
                                columns_to_select = ['playerFullName', 'Age', 'Team', 'Position', f'{metric}_max', f'{metric}_avg_over_selected']
                                metric_display_name = f'{metric}_max'
                            else:
                                columns_to_select = ['playerFullName', 'Age', 'Team', 'Position', metric, f'{metric}_cum_avg']
                                metric_display_name = metric

                            available_columns = [col for col in columns_to_select if col in latest_data.columns]
                            top10 = latest_data[available_columns].dropna(subset=[metric_display_name]).sort_values(by=metric_display_name, ascending=False).head(10)

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
                                top10.rename(columns={'playerFullName': 'Player'}, inplace=True)

                                # Format the metric value with the appropriate averages
                                if metric == 'PSV-99':
                                    top10[metric] = top10.apply(
                                        lambda row: f"{row[f'{metric}_max']:.2f} ({row[f'{metric}_avg_over_selected']:.2f})" if pd.notnull(row[f'{metric}_avg_over_selected']) else f"{row[f'{metric}_max']:.2f}",
                                        axis=1
                                    )
                                    # Remove the extra columns
                                    top10.drop(columns=[f'{metric}_max', f'{metric}_avg_over_selected'], inplace=True)
                                else:
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

                                    # Prepare the aggregation dictionary using named aggregations
                                    agg_dict_overall = {
                                        'Age': 'last',
                                    }

                                    if team_column:
                                        agg_dict_overall['Team'] = (team_column, 'last')
                                    if position_column in metric_data_overall.columns:
                                        agg_dict_overall['Position'] = (position_column, 'last')

                                    agg_dict_overall[f'{metric}_max'] = (metric, 'max')
                                    agg_dict_overall[f'{metric}_avg_over_selected'] = (metric, 'mean')

                                    latest_data_overall = metric_data_overall.groupby('playerFullName').agg(**agg_dict_overall).reset_index()

                                    # Round the Age column to ensure no decimals
                                    latest_data_overall['Age'] = latest_data_overall['Age'].round(0).astype(int)

                                    # Prepare the data
                                    columns_to_select_overall = ['playerFullName', 'Age', 'Team', 'Position', f'{metric}_max', f'{metric}_avg_over_selected']
                                    available_columns_overall = [col for col in columns_to_select_overall if col in latest_data_overall.columns]
                                    top10_overall = latest_data_overall[available_columns_overall].dropna(subset=[f'{metric}_max']).sort_values(by=f'{metric}_max', ascending=False).head(10)

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
                                        top10_overall.rename(columns={'playerFullName': 'Player'}, inplace=True)

                                        # Format the metric value with average over selected matchdays
                                        top10_overall[metric] = top10_overall.apply(
                                            lambda row: f"{row[f'{metric}_max']:.2f} ({row[f'{metric}_avg_over_selected']:.2f})" if pd.notnull(row[f'{metric}_avg_over_selected']) else f"{row[f'{metric}_max']:.2f}",
                                            axis=1
                                        )

                                        # Remove the extra columns
                                        top10_overall.drop(columns=[f'{metric}_max', f'{metric}_avg_over_selected'], inplace=True)

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
                # [ ... existing code for glossary ... ]

        else:
            st.write("Please set your filters and click 'Run' to display the data.")
