import plotly.express as px
import pandas as pd

df = pd.read_csv("elonmusk.csv")

df
data_canada = px.data.gapminder().query("country == 'Canada'")
print(data_canada, type(data_canada))
fig = px.bar(data_canada, x='year', y='pop')
fig.show()