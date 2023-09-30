#from streamlit_pandas_profiling import st_profile_report
#from ydata_profiling import ProfileReport
import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from sklearn.datasets import load_iris
from ydata_profiling.utils.cache import cache_file

st.set_page_config(layout="wide")
st.title('Iris Flowers - Classification')
st.caption('Created by Bayhaqy')
st.markdown('Classify iris flowers into \
setosa, versicolor, virginica')

st.image('https://machinelearninghd.com/wp-content/uploads/2021/03/iris-dataset.png')
st.image('https://www.integratedots.com/wp-content/uploads/2019/06/iris_petal-sepal-e1560211020463.png')

# Load Dataset
#iris = load_iris(as_frame=True)

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    return df

iris = cache_file(
  'Iris.csv',
  'https://raw.githubusercontent.com/bayhaqy/Classification-Iris-Prediction/main/Iris.csv',
)

df = load_data(iris)

# Create a DataFrame from the iris data
#df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add a target column to the DataFrame
#df['Target'] = iris['target']

# Translate the target
#df['Target'] = df['Target'].apply(lambda x: iris['target_names'][x])

st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
  st.text('Sepal Size')
  sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
  sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)

with col2:
  st.text('Pepal Size')
  petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
  petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)

if st.button('Predict type of Iris'):
  result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
  st.text(result[0])

st.write("---")
if st.checkbox("Sample rendom the Iris Dataset"):
    st.write(df.sample(10))  # Same as st.write(df)
    #pr = ProfileReport(df,title="Dataset Report",correlations=None)
    #st_profile_report(pr)

st.write("---")
