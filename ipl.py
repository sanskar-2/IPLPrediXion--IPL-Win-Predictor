import base64
import streamlit as st
import pickle
import pandas as pd

# st.markdown(
#     """
#     <h1 style="color: white; text-align: center;">IPL Win Predictor</h1>
#     """,
#     unsafe_allow_html=True
# )

# Add custom CSS for font color
st.markdown(
    """
    <style>
    /* Change font color for all input labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: white; /* Set the font color to white */
        font-size: 16px; /* Optional: Adjust font size */
        font-weight: bold; /* Optional: Make text bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to convert image to Base64
def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Convert your image to Base64
img = get_img_as_base64("ipl-2020.jpg")

# Add background image using Streamlit's HTML style injection
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/png;base64,{img}");
background-size: cover;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# IPL Win Predictor Code
teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl','rb'))
st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1,
                           help="Maximum 20 overs allowed")
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1,
                             help="Maximum 10 wickets allowed")




if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.markdown(
        f"""
            <h2 style="color: white; text-align: center;">{batting_team} - {round(win * 100)}%</h2>
            <h2 style="color: white; text-align: center;">{bowling_team} - {round(loss * 100)}%</h2>
            """,
        unsafe_allow_html=True
    )
