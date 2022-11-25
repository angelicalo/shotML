import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title='Ford Preços',
    page_icon=":car:"
)

with open('lin_reg_ford.pickle', 'rb') as model:
    reg = pickle.load(model)

cols = ['year', 'mileage', 'model_ B-MAX', 'model_ C-MAX', 'model_ EcoSport',
        'model_ Edge', 'model_ Fiesta', 'model_ Focus', 'model_ Galaxy',
        'model_ Grand C-MAX', 'model_ KA', 'model_ Ka+', 'model_ Kuga',
        'model_ Mondeo', 'model_ S-MAX', 'model_Outros',
        'transmission_Automatic', 'transmission_Manual',
        'transmission_Semi-Auto', 'fuelType_Diesel', 'fuelType_Electric',
        'fuelType_Hybrid', 'fuelType_Other', 'fuelType_Petrol']

car_user = pd.DataFrame(np.zeros((1,24)),index=[0],columns=cols)
    
model = st.selectbox('Qual modelo do seu carro?', (None, ' Fiesta', ' Focus', 'Outros', ' Kuga', ' EcoSport', ' C-MAX',\
                                                   ' Mondeo', ' Ka+', ' S-MAX', ' B-MAX', ' Edge', ' Grand C-MAX',' KA', ' Galaxy'))

transmission = st.selectbox('Qual o tipo de transmissão do seu carro?', (None, 'Automatic', 'Manual', 'Semi-Auto'))

fuelType = st.selectbox('Qual o tipo de combustível do seu carro?', (None, 'Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'))

year = st.number_input('Qual o ano do seu carro?', min_value=2000, max_value=2022, value=2022, step=1)

mileage = st.number_input('Quantas milhas já rodou o seu carro?', min_value=0, max_value=None, value=0, step=None)


char_index = ['Modelo', 'Transmissão', 'Combustível', 'Ano', 'Milhas']
char = [model, transmission, fuelType, year, mileage]

if None in char:
    st.write('Preciso de todas as características solicitadas para tentar acertar qual é o seu animal')
else:
    car_user['model_'+ model] = 1
    car_user['transmission_' + transmission] = 1
    car_user['fuelType_'+ fuelType] = 1
    car_user['year'] = year
    car_user['mileage'] = mileage
    
    df = pd.DataFrame(car_user, index = [0])
    # st.dataframe(df)
    df = pd.DataFrame(char, index =char_index )
    st.dataframe(df.transpose())
    
    price_pred = reg.predict(np.array(car_user).reshape(1,-1))
    
    st.write(f'Sugerimos o preço de venda para o seu carro sendo R${np.round(price_pred[0],2)}')                 
