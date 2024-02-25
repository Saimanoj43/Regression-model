###### Libraries #######

# Base Libraries
import pandas as pd
import numpy as np

# Deployment Library
import streamlit as st

# Model Pickled File Library
import joblib

############# Data File ###########

data = pd.read_csv("hotelfinal")

data = data.dropna(axis=0).reset_index(drop=True)

########### Loading Trained Model Files ########
model = joblib.load("Hotel_price.pkl")
model_ohe = joblib.load("Hotel_ohe.pkl")
model_sc = joblib.load("Hotel_sc.pkl")

########## UI Code #############

# Ref: https://docs.streamlit.io/library/api-reference

# Title
st.title("$ Estimation of hotel room price for the Given Person Details:")

# Image
with st.columns(5)[1]:
    st.image("https://png.pngtree.com/png-clipart/20230921/original/pngtree-hotel-booking-word-concepts-banner-trip-planning-amenities-reviews-vector-png-image_12474719.png", width=400)

# Description
st.write("""Built a Predictive model in Machine Learning to estimate the hotel price for a person can get.
         Sample Data taken as below shown.
""")

# Data Display
del data['Unnamed: 0']
del data['hotel_name']
del data['location']
del data['price']
del data['taxes']
del data['food_price']

data.guests = data.guests.str.strip(' x Guests') # removing the 'Guests' from the guests column
data.guests = data.guests.astype(float)
data.room = data.room.str.strip(' x Rooms') # Removing the ' x Rooms' from the room column
data.room = data.room.astype(float)

        
data.rename(columns={'additional_info':'breakfast_Included'}, inplace=True)
for i in range(len(data)):
    try:
        if 'Free Cancellation Till' in data.breakfast_Included[i]:
            data.breakfast_Included[i]="no"
        elif'Free Breakfast Included' == data.breakfast_Included[i]:
            data.breakfast_Included[i]="yes"
    except:
        continue

st.dataframe(data.head())
st.write("From the above data , Price is the prediction variable")

###### Taking User Inputs #########
st.subheader("Enter Below Details to Get the Estimation Prices:")

col1, col2, col3 = st.columns(3) # value inside brace defines the number of splits
col4, col5, col6 = st.columns(3)
col7, col8, col9 = st.columns(3) # value inside brace defines the number of splits
col10, col11= st.columns(2)


with col1:
    city = st.selectbox("Enter City Name:",data.city.unique())
    st.write(city)

with col2:
    breakfast_Included = st.radio("Enter Breakfast Required or not :",data.breakfast_Included.unique())
    st.write(breakfast_Included)


with col3:
    star_rating = st.number_input("Enter type hotel in terms of star(1 to 5 scale):",)
    st.write(star_rating)

with col4:
    couple = st.radio("Enter Married couple or not : ",data.couple.unique())
    st.write(couple)

with col5:
    pets = st.radio("If you have any pets to stay:", data.pets.unique())
    st.write(pets)

with col6:
    rating = st.number_input("Enter rating of the Hotel(0 to 5.0 scale) :  ")
    st.write(rating)

with col7:
    no_of_ratings = st.number_input("Enter Minimum no.of rating count you required (Int): ",)
    st.write(no_of_ratings)

with col8:
    guests = st.number_input("Enter Number Of Guests (Int) : ")
    st.write(guests)

with col9:
    room = st.number_input("No.of rooms required (Int):")
    st.write(room)

with col10:
    check_in = st.time_input("Enter check-in time (format : 1:30 AM) :  ")
    st.write(check_in)

with col11:
    check_out = st.time_input("Enter check-out time (format : 1:30 AM) : ")
    st.write(check_out)



###### Predictions #########

if st.button("Estimate Price"):
    st.write("Data Given:")
    values = [city, breakfast_Included,star_rating,couple,pets,rating,no_of_ratings,guests,room,check_in,check_out]
    record =  pd.DataFrame([values],
                           columns = ['city', 'breakfast_Included',
                           'star_rating','couple','pets','rating','no_of_ratings',
                           'guests','room','check_in','check_out'])
    
    st.dataframe(record)

    # 'breakfast_Included' , 'couple', 'pets'
    record.breakfast_Included.replace({'yes':1, 'no':0},inplace=True)
    record.couple.replace({'yes':1, 'no':0,'No':0},inplace=True)
    record.pets.replace({'yes':1, 'no':0},inplace=True)

    record.check_in = record.check_in.astype(str)
    # Converting the hours and minutes data into minutes of the col
    for i in range(len(record)):
        if ':' in record['check_in'][i]:
            record["check_in"][i] = (60*(int(record['check_in'][i].split(':')[0])))+int(record['check_in'][i].split(':')[1])
        
    record.check_in = record.check_in.astype(int)


    record.check_out = record.check_out.astype(str)
    # Converting the hours and minutes data into minutes of the col
    for i in range(len(record)):
        if ':' in record['check_out'][i]:
            record["check_out"][i] = (60*(int(record['check_out'][i].split(':')[0])))+int(record['check_out'][i].split(':')[1])

    record.check_out = record.check_out.astype(int)

        
    ohedata = model_ohe.transform(record.iloc[:,[0]]).toarray()

    ohedata = pd.DataFrame(ohedata, columns = model_ohe.get_feature_names_out())

    record = pd.concat([record.iloc[:,1:], ohedata], axis = 1)
    
    #scaling
    record.iloc[:, [5,8,9]] = model_sc.transform(record.iloc[:, [5,8,9]])

    # st.dataframe(record)
    charges = round(model.predict(record)[0],2)
    charges = str(charges)+" ₹"
    st.subheader("Estimated Charges:")
    st.subheader(charges)
