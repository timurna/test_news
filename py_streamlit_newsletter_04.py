import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer

# Set the page configuration to wide mode
st.set_page_config(layout="wide")

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

# Load the dataset from Parquet
file_path = 'https://raw.githubusercontent.com/timurna/test_news/main/newupclean3.parquet'
data = pd.read_parquet(file_path)

# Ensure 'League' column values are consistent by stripping any leading/trailing whitespaces
data['League'] = data['League'].str.strip()

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

# Create a single row for all the filters
with st.container():
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        leagues = sorted(data['League'].unique())  # Sort leagues alphabetically
        
        # Debugging: Display unique leagues to verify presence of "UEFA Champions League (Europe)"
        st.write("Available Leagues:", leagues)
        
        selected_league = st.selectbox("Select League", leagues, key="select_league")

    with col2:
        # Filter data based on the selected league
        league_data = data[data['League'] == selected_league]

        # Week Summary and Matchday Filtering Logic
        week_summary = league_data.groupby(['League', 'Week']).agg({'Date': ['min', 'max']}).reset_index()
        week_summary.columns = ['League', 'Week', 'min', 'max']

        week_summary['min'] = pd.to_datetime(week_summary['min'])
        week_summary['max'] = pd.to_datetime(week_summary['max'])

        week_summary['Matchday'] = week_summary.apply(
            lambda row: f"{row['Week']} ({row['min'].strftime('%d.%m.%Y')} - {row['max'].strftime('%d.%m.%Y')})", axis=1
        )

        # Sort weeks by the minimum date and remove duplicates based on 'Week'
        filtered_weeks = week_summary[week_summary['League'] == selected_league].sort_values(by='min').drop_duplicates(subset=['Week'])

        matchday_options = filtered_weeks['Matchday'].tolist()
        selected_matchday = st.selectbox("Select Matchday", matchday_options, key="select_matchday")

        selected_week = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['Week'].values[0]

        # Get the date of the selected matchday
        selected_date = filtered_weeks[filtered_weeks['Matchday'] == selected_matchday]['max'].values[0]

    with col3:
        position_group_options = list(position_groups.keys())
        selected_position_group = st.selectbox("Select Position Group", position_group_options, key="select_position_group")

# Now you can continue your application logic using `selected_league`, `selected_date`, `selected_position_group`.

# Set the custom CSS
if st.session_state.authenticated:
    set_mobile_css()
