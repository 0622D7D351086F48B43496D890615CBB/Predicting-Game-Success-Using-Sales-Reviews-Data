import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

DATA_PATH = "data/all_games.csv"
df = pd.read_csv(DATA_PATH, encoding="latin1")
df['meta_score'] = pd.to_numeric(df['meta_score'], errors='coerce')
df['user_review'] = pd.to_numeric(df['user_review'], errors='coerce')
df.dropna(subset=['meta_score','user_review'], inplace=True)
df['hit_flop'] = df['meta_score'].apply(lambda x: 'Hit' if x >= 75 else 'Flop')

app = Dash(__name__)
app.title = "Game Success Dashboard"

app.layout = html.Div(children=[
    html.H1("Game Success Dashboard", style={'textAlign':'center'}),
    
    html.H2("Platform-wise Average Meta Score"),
    dcc.Graph(
        figure=px.bar(
            df.groupby('platform')['meta_score'].mean().reset_index(),
            x='platform', y='meta_score', color='meta_score',
            color_continuous_scale='Viridis', title="Average Meta Score by Platform"
        )
    ),

    html.H2("Meta Score vs User Review"),
    dcc.Graph(
        figure=px.scatter(
            df, x='meta_score', y='user_review', color='platform',
            hover_data=['name'], title="Meta Score vs User Review"
        )
    ),

    html.H2("Hit vs Flop Games"),
    dcc.Graph(
        figure=px.pie(df, names='hit_flop', title="Hit vs Flop Games")
    ),
])

if __name__ == "__main__":
    app.run(debug=True)  # Dash 3+ syntax
