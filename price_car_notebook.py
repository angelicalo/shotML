# !pip install autoviz
# !pip install evidently
# !pip install --pre pycaret
# !pip install streamlit
# # !pip install  pandas-profiling

# Commented out IPython magic to ensure Python compatibility.
from autoviz.AutoViz_Class import AutoViz_Class
from pandas_profiling import ProfileReport
from fastai.tabular.core import cont_cat_split
from yellowbrick.classifier import confusion_matrix
from yellowbrick.classifier import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,\
                                    GridSearchCV
from sklearn.preprocessing import (StandardScaler, 
                                   OneHotEncoder)
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn import metrics

import pandas as pd
import numpy as np

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

import plotly.express as px

# ML 03
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab

from pycaret.datasets import get_data
from pycaret import regression
from pycaret import arules

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import pickle 

# %matplotlib inline

"""# Propósito do modelo preditivo (need statement)

Este modelo tem a função de auxiliar nossos clientes na revenda de seus carros. 

A base de dados utilizada está disponível em: https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes?resource=download&select=ford.csv

O conjunto de dados contém informações de preço, transmissão, quilometragem, tipo de combustível, imposto rodoviário, milhas por galão (mpg) e tamanho do motor.

# Análise exploratória de dados (gráfico e tabela)

Nesta seção, seão apresentados alguns gráficos e tabelas a fim de explorar o conjunto de dados.
"""

# data = pd.read_csv('ford.csv')
data = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv')

profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_file("profile.html")

data.describe() # algumas estatísticas básicas para os dados numéricos

cont_names, cat_names = cont_cat_split(data)
for c in ['year','mileage','price']:
    fig = px.histogram(data[cont_names], x=c, marginal="box")
    fig.update_layout(title_text=f"Histogram for {c}")
    fig.show()
    fig = px.histogram(data[cont_names], x=c,color='year')
    fig.show()

fig = px.histogram(data[cat_names], x='model', color="model")
fig.update_xaxes(type='category')
plt.savefig('barmodel.jpg')
fig.show()

count_model = pd.DataFrame(data['model'].value_counts())
count_model

text = " ".join(w for w in data['model'].values)
word_cloud = WordCloud(
        width=3000,
        height=2000,
        random_state=1,
        background_color="salmon",
        colormap="Pastel1",
        collocations=False,
        stopwords=STOPWORDS,
        ).generate(text)
plt.figure(figsize=(10,8))
plt.imshow(word_cloud)
plt.axis("off")
plt.savefig('wordcloud.jpg')
plt.show()

top5 = count_model.head(5).index
plt.figure(figsize=(10,8))
# top5 carros e densidade dos preços
data.loc[data['model'].isin(top5)].groupby('model')['price'].plot(kind='density',legend=True)
plt.show()

"""

# Engenharia de features e tratamento dos dados

"""

AV = AutoViz_Class()
dft = AV.AutoViz(filename="", dfte=data, chart_format='html')

tail = count_model.tail(10).index
tail

# combinando categorias raras
outros = len(tail)*['Outros']
dic = dict(zip(tail,outros))
data['model'] = data['model'].replace(dic,regex=True)
pd.DataFrame(data['model'].value_counts())

# transformando os dados categóricos e normalizando os numéricos
y = data['price']
X = data.drop(columns=['price'])

# dados numericas e categoricas
cont_names, cat_names = cont_cat_split(X)

# separando em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# categorical_transformer
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
dt = categorical_transformer.fit_transform(X_train[cat_names]).toarray()
cols = categorical_transformer.get_feature_names_out()
X_train_cat = pd.DataFrame(dt, columns=cols)
X_train_cat['fuelType_Electric'] = 0

dt = categorical_transformer.fit_transform(X_test[cat_names]).toarray()
cols = categorical_transformer.get_feature_names_out()
X_test_cat = pd.DataFrame(dt, columns=cols)
X_test_cat['fuelType_Other'] = 0
X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns)

# numeric_transformer
scaler = StandardScaler().fit(X_train[cont_names])
X_train_num = pd.DataFrame(scaler.fit_transform(X_train[cont_names]), columns=cont_names)

scaler = StandardScaler().fit(X_test[cont_names])
X_test_num = pd.DataFrame(scaler.fit_transform(X_test[cont_names]), columns=cont_names)

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# # outra forma - transformando os dados categóricos e normalizando os numéricos
# y = data['price']
# X = data.drop(columns=['price'])

# # dados numericas e categoricas
# cont_names, cat_names = cont_cat_split(X)

# # separando em treino e teste (70/30)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# numeric_transformer = Pipeline(
#     steps=[("imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler())])

# categorical_transformer = Pipeline(
#     steps=[("ohe", OneHotEncoder(handle_unknown="ignore", drop='first'))])
    
# preprocessor = ColumnTransformer(transformers=[ 
#         ("num", numeric_transformer, cont_names),
#         ("cat", categorical_transformer, cat_names)])

# X_train = preprocessor.fit_transform(X_train) 
# X_test = preprocessor.fit_transform(X_test) 
# len(X_train), X_train

"""

# Data drift (sugestão, utilizar o evidently)"""

dashboard = Dashboard(tabs=[DataDriftTab(verbose_level=0)])
dashboard.calculate(X_train, X_test, column_mapping=None)
dashboard.show(mode='inline')

# target drift
y_train_target = pd.DataFrame(data=y_train).rename(columns={'price':'target'})
y_test_target = pd.DataFrame(data=y_test).rename(columns={'price':'target'})

dashboard = Dashboard(tabs=[CatTargetDriftTab(verbose_level=0)])
dashboard.calculate(y_train_target, y_test_target)
dashboard.show(mode='inline')



"""# Construção do modelo preditivo e principais métricas

De acordo com o Pandas Profiling Report, observa-se que o preço tem alta correlação com as seguintes características: 

$$'model',\, 'year',\, 'transmission',\, 'mileage',\, 'fuelType'$$
"""

# seleção de feature para criação do modelo preditivo
features, target = ['model', 'year', 'transmission', 'mileage', 'fuelType'],['price']

X = data[features]
y = data[target]

Xy = X.copy()
Xy['price'] = y
Xy

# apois para encontar melhor modelo preditivo para o conjunto de dados
s = regression.setup(Xy, target = 'price', session_id=123)
best = regression.compare_models()

# o melhor modelo foi o gbr, porem com ele o streamlit não estava dando certo a aplicação. 
# Com isso o valor de R^2 diminuiu 
lreg = regression.create_model('lr')
regression.plot_model(lreg, plot='residuals')
regression.plot_model(lreg, plot='error')
with open('lr_ford.pickle', 'wb') as model:
    pickle.dump(lreg, model)

# tratamento das variáveis categóricas
y = Xy['price']
X = Xy.drop(columns=['price'])

# dados numericas e categoricas
cont_names, cat_names = cont_cat_split(X)

# categorical_transformer
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
dt = categorical_transformer.fit_transform(X[cat_names]).toarray()
X_cat = pd.DataFrame(dt, columns=categorical_transformer.get_feature_names_out())

X_num = pd.DataFrame(X[cont_names], columns=cont_names)

X = pd.concat([X_num, X_cat], axis=1)

# separando em treino e teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)

with open('lin_reg_ford.pickle', 'wb') as model:
    pickle.dump(reg, model)

# !pip install matplotlib==3.1.3
n = 30
y_pred = reg.predict(X_test.iloc[:n])
x = range(1,n+1)
plt.plot(x,y_test.iloc[:n],label='test')
plt.plot(x,y_pred,label='train')
plt.title(f'Preço dos {n} primeiros carros de teste')
if len(x) <= 20:
    plt.xticks(x, x)
plt.legend()
plt.show()

y.iloc[0], X.iloc[0] # valor usado no teste do streamlit
