import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
st.set_page_config(layout="wide")
scaler = StandardScaler()
st.title("Resturant review system")


st.caption("Your Guide to Culinary Experiences: Discover, Rate, and Share Restaurant Reviews!")
st.divider()

averagecost = st.number_input("please estimate the average cost of two",min_value=50,max_value=999999,value=1000,step=200)
tablebooking = st.selectbox("Resturant has table booking?",["yes","no"])
onlinedelivering = st.selectbox("Resturant has online delivering",["yes","no"])
pricerange = st.selectbox("what is the price range (1 cheapest 4 most expensive)",[1,2,3,4])
pricebutton = st.button("please review!")
st.divider()
model = joblib.load("ml_model.pkl")

bookingstatus = 1 if tablebooking =="yes" else 0
deliveringstatus = 1 if onlinedelivering == "yes" else 0

averagecost = scaler.fit_transform(averagecost)
bookingstatus = scaler.fit_transform(bookingstatus)
deliveringstatus =scaler.fit_transform(deliveringstatus)
pricerange = scaler.fit_transform(pricerange)

