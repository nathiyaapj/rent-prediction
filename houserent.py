import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from xgboost import XGBRegressor
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from PIL import Image

st.set_page_config(page_title="House_Rent_Predicting | By Karthik ",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This dashboard app is created !
                                        Data has been gathered from machine learning model"""}
                   )
# Load data using st.cache_data


@st.cache_data
def load_data():
    # Load data
    data = pd.read_csv('')
    sample_df = data.sample(n=5000, random_state=72)
    return sample_df


df = load_data()


# Train model using st.cache_data
@st.cache_data
def train_model(data):
    # Train model here
    x = data.drop('rent', axis=1)
    y = data['rent']
    trained_model = XGBRegressor()
    trained_model.fit(x, y)
    return trained_model


xgregressor = train_model(df)


selected = option_menu(
    menu_title=None,
    options=["Info", "Predict", "Contact"],
    icons=["book", "stopwatch", "toggles"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "white", "size": "cover", "width": "100%"},
            "icon": {"color": "black", "font-size": "20px"},
            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "#6F36AD"},
            "nav-link-selected": {"background-color": "#6F36AD"}})
if selected == "Info":
    if st.button('Click to view data information'):
        st.header('**Input DataFrame**')
        st.write(df)
        st.header('**Pandas Profiling Report**')
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)


if selected == "Predict":
    def main():
        st.sidebar.title('Please provides the required details')
        st.title('House_rent_prediction')
        type1 = {'BHK1': 1, 'BHK2': 2, 'BHK3': 3, 'BHK4PLUS': 4, 'RK1': 5}
        type_key = st.sidebar.selectbox('Select a TYPE', list(type1.keys()))
        type = type1[type_key]

        st.sidebar.title('Please provides the required details')

        lease_type1 = {'FAMILY': 1, 'ANYONE': 2, 'BACHELOR': 3, 'COMPANY': 4}
        lease_key = st.sidebar.selectbox('Select a LEASE', list(lease_type1.keys()))
        lease_type = lease_type1[lease_key]

        st.sidebar.title('Please provides the required details')

        parking1 = {'BOTH': 1, 'TWO_WHEELER': 2, 'FOUR_WHEELER': 3, 'NONE': 4}
        parking_key = st.sidebar.selectbox('Select a PARKING', list(parking1.keys()))
        parking = parking1[parking_key]

        st.sidebar.title('Please provides the required details')

        facing1 = {'E': 1, 'N': 2, 'S': 3, 'W': 4, 'SE': 5,'NE': 6,'SW': 7,'NW': 8}
        facing_key = st.sidebar.selectbox('Select a FACE', list(facing1.keys()))
        facing = facing1[facing_key]

        st.sidebar.title('Please provides the required details')

        water_supply1 = {'CORP_BORE': 1, 'CORPORATION': 2, 'BOREWELL': 3}
        water_supply_key = st.sidebar.selectbox('Select a WATER_SUPPLY', list(water_supply1.keys()))
        water_supply = water_supply1[water_supply_key]

        st.sidebar.title('Please provides the required details')

        building_type1 = {'IF': 1, 'AP': 2, 'IH': 3, 'GC': 4}
        building_type_key = st.sidebar.selectbox('Select a BULDING_TYPE', list(building_type1.keys()))
        building_type = building_type1[building_type_key]







        property_size = st.sidebar.number_input("Enter the area", value=1, min_value=1)
        bathroom= st.sidebar.number_input("Enter the bathroom", value=0)
        property_age= st.sidebar.number_input("Enter the bathroom", value=1, min_value=1)
        floor = st.sidebar.number_input("Enter the bathroom", value=1.0, min_value=1.0, max_value=None, step=1.0)

        total_floor = st.sidebar.number_input("Enter the bathroom", value=1, min_value=1, max_value=None, step=1, key="total_floor_input")

        balconies = st.sidebar.number_input("Enter the bathroom", value=1, min_value=1, max_value=None, step=1, key="balconies_input")


        latitude = st.sidebar.number_input("Enter the lower bound of the storey range", value=4, min_value=0)
        longitude = st.sidebar.number_input("Enter the upper bound of the storey range", value=6,
                                                     min_value=latitude)
        gym = st.sidebar.number_input("Enter the bathroom", value=0, key="gym_widget")




        # create dictionary for features
        features = {'type': type,
                    'lease_type': lease_type,
                    'parking': parking,
                    'facing': facing,
                    'water_supply': water_supply,
                    'building_type': building_type,
                    'property_size': property_size,
                    'bathroom': bathroom,
                    'property_age': property_age,
                    'floor': floor,
                    'total_floor': total_floor,
                    'balconies': balconies,
                    'latitude': latitude,
                    'longitude': longitude,
                    'gym':gym,

                    }

        features_df = pd.DataFrame(features, index=[1])
        # create dataframe using the collected features
        st.dataframe(features_df)
        # predict the resale price
        if st.button('Predict'):
            # Use the trained model to make predictions
            prediction = xgregressor.predict(features_df)  # Replace X_test with your test data

            # Display the prediction
            st.write("Rent Price:", prediction)


    if __name__ == '__main__':
        main()

if selected == "Contact":
    name = "nathiyaapj"
    mail = (f'{"Mail :"}  {"nathiyapalanisamy72@gmail.com"}')
    description = "An Aspiring DATA-SCIENTIST..!"
    social_media = {"GITHUB": "https://github.com/nathiyaapj",
        "LINKEDIN": " www.linkedin.com/in/ nathiya-palanisamy-610805285 "
   }
