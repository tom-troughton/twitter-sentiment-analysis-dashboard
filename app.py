from inspect import stack
from sre_parse import State
from tempfile import tempdir
from tkinter import N
from turtle import bgcolor, color, fillcolor, width
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq
from matplotlib.pyplot import autoscale
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import tweepy
from tweepy import OAuthHandler
import time
from datetime import datetime, timedelta
import requests
from requests.api import request
import random
import credentials
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import pandas as pd
import numpy as np
import pytz
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
from io import BytesIO
import base64

# Twitter API Authentication
# API keys are imported from credentials.py
auth = OAuthHandler(credentials.CONSUMER_KEY, credentials.CONSUMER_SECRET_KEY)
auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

client = tweepy.Client(credentials.BEARER_TOKEN, wait_on_rate_limit=True)


# --- Setting some variables ---
# Number of days back to search for tweets. Min: 1, Max: 7
num_days = 7
# Number of tweets to retrieve per day. Min: 10, Max: 100
num_tweets = 100

# Colours
bg_color = '#303030'
color_green = '#3eb869'
color_orange = '#d6993c'
color_red = '#e35970'

# Figure dimensions
fig_width_1 = 305
fig_width_2 = 630
fig_height = 305

# Custom style for plotly figures
pio.templates['custom'] = go.layout.Template(
    layout = {
        'title':
            {'font': 
                {
                'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif',
                'size':30,
                'color': '#FFFFFF'
                }
            },
        'font': 
            {
            'family': 'Helvetica Neue, Helvetica, Sans-serif', 
            'size':16,
            'color': '#FFFFFF'
            },
        'paper_bgcolor': bg_color,
        'plot_bgcolor': bg_color,
        'polar': {'bgcolor': bg_color},
        'xaxis': {'showgrid': False, 'showline': True},
        'yaxis': {'showgrid': False, 'showline': True}
    }
)


# *********************************** TWEET RETRIEVAL & SENTIMENT ANALYSIS ***********************************

# Code for retrieving embedding code to embed tweets on the dashboard
# def get_embed(tweet_url):
#     url = 'https://publish.twitter.com/oembed?url=' + tweet_url
#     response = requests.get(url)
#     html = response.json()
#     return html['html']

def get_embed(df, sentiment):
    if sentiment == 'positive':
        sample_pos_tweet_url = 'https://twitter.com/twitter/status/' + str(df.loc[df['compound'] > 0].sort_values(by='compound', ascending=False).iloc[0].tweet_id)

        url = 'https://publish.twitter.com/oembed?url=' + sample_pos_tweet_url
        response = requests.get(url)
        html = response.json()
        return html['html']

    elif sentiment == 'negative':
        sample_neg_tweet_url = 'https://twitter.com/twitter/status/' + str(df.loc[df['compound'] < 0].sort_values(by='compound', ascending=True).iloc[0].tweet_id)

        url = 'https://publish.twitter.com/oembed?url=' + sample_neg_tweet_url
        response = requests.get(url)
        html = response.json()
        return html['html']

# Removing #, @, etc. from tweets
def clean_tweet(tweet):
    # Removing links
    tweet = re.sub(r"(http?\://|https?\://|www)\S+", "", tweet)

    # Removing twitter handles. Regex code credit: user: 'Negi Babu', https://stackoverflow.com/questions/50830214/remove-usernames-from-twitter-data-using-python
    tweet = re.sub('@[^\s]+', '', tweet)

    # Removing hashtag sign, but keeping the hashtag text
    tweet = tweet.replace('#', '')

    # Removing 'amp'. There are some occurences of tweets containing this, perhaps due to encoding problems
    tweet = tweet.replace('amp', '')

    return tweet

# Function to retrieve tweets via tweepy
def retrieve_tweets(query, num_days, num_tweets, model):
    # Creating empty dictionary. This will be used to create dataframe.
    data = {'tweet_id': [], 'tweet': [], 'date': [], 'negative': [], 'neutral': [], 'positive': [], 'compound': [], 'model': []}

    date = datetime.now() + timedelta(days = 1) - timedelta(minutes = 1)

    # Converting timezone to UTC for the Twitter API
    date = date.astimezone(pytz.utc)

    for i in range(num_days):
        date -= timedelta(days = 1)

        # Adding filters to the query. Prevents retrieving retweets and tweets which contain media or links. Retrieves only english language tweets.
        query = query + ' -is:retweet -has:media -has:links lang:en'
        
        # Retrieving tweets from the past 7 days. Collecting 100 tweets for each day.
        tweets = client.search_recent_tweets(
                                        query, 
                                        tweet_fields=['created_at', 'text', 'author_id'], 
                                        expansions='author_id', 
                                        max_results=num_tweets, 
                                        start_time=((date - timedelta(days=1)) + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ"), 
                                        end_time=date.strftime("%Y-%m-%dT%H:%M:%SZ"), 
                                        sort_order='relevancy'
                                        )
        
        if model.lower() == 'vader':
            sia = SentimentIntensityAnalyzer()
            for tweet in tweets[0]:
                data['date'].append(tweet.created_at)
                data['tweet_id'].append(tweet.id)
                data['tweet'].append(tweet.text)

                # Performing sentiment analysis on the tweets. Adding the scores to dictionary.
                # Text preprocessing is not necessary with VADER model.
                scores = sia.polarity_scores(tweet.text)
                data['negative'].append(scores['neg'])
                data['neutral'].append(scores['neu'])
                data['positive'].append(scores['pos'])
                data['compound'].append(scores['compound'])
                data['model'].append('vader')

        elif model.lower() == 'textblob':
            for tweet in tweets[0]:
                data['date'].append(tweet.created_at)
                data['tweet_id'].append(tweet.id)
                data['tweet'].append(tweet.text)

                scores = TextBlob(tweet.text).sentiment.polarity
                data['negative'].append(np.nan)
                data['neutral'].append(np.nan)
                data['positive'].append(np.nan)
                data['compound'].append(scores)
                data['model'].append('textblob')

    # Creating dataframe from tweet data
    df = pd.DataFrame(data)

    # Creating a column with shorter date format
    df['date_short'] = df['date'].apply(lambda x: x.date())

    # Creating new column for cleaned tweets
    df['clean_tweet'] = df['tweet'].apply(lambda x: clean_tweet(x))

    return df

# Return percentages of positive, neutral and negative tweets in a given dataframe
def sentiment_percent(df):
    pos_pct = round(((df.loc[df['compound'] > 0].count()[0]/df.shape[0]) * 100))
    neu_pct = round(((df.loc[df['compound'] == 0].count()[0]/df.shape[0]) * 100))
    neg_pct = round(((df.loc[df['compound'] < 0].count()[0]/df.shape[0]) * 100))

    return pos_pct, neu_pct, neg_pct

# Function for retrieving the number of tweets containing the hashtag
def number_tweets(query):
    date = datetime.now() - timedelta(minutes = 1)
    date = date.astimezone(pytz.utc)

    tweet_count = client.get_recent_tweets_count(
                                        query,
                                        granularity='day'
                                        )
    
    return tweet_count


# *********************************** CREATING FIGURES ***********************************

# Pie chart containing only positive and negative sentiment
def pie_chart1(df):
    pos_tweets = df.loc[df['compound'] > 0].count()[0]
    neg_tweets = df.loc[df['compound'] < 0].count()[0]

    fig = go.Figure(
                    data = [go.Pie(
                                labels = ['positive', 'negative'], 
                                values = [pos_tweets, neg_tweets], 
                                hole = 0.6
                                )
                            ]
                    )

    fig.update_traces(
                hoverinfo = 'label', 
                marker = dict(colors=['#3eb869', '#e35970']), 
                textposition = 'inside'
                )

    fig.update_layout(
                template = 'custom', 
                showlegend = False, 
                margin = dict(t=20, b=20, l=20, r=20)
                )

    return fig

# Pie chart containing positive, neutral, and negative sentiment
def pie_chart2(df):
    pos_tweets = df.loc[df['compound'] > 0].count()[0]
    neu_tweets = df.loc[df['compound'] == 0].count()[0]
    neg_tweets = df.loc[df['compound'] < 0].count()[0]

    fig = go.Figure(
                    data = [go.Pie(
                                labels = ['positive', 'neutral', 'negative'], 
                                values = [pos_tweets, neu_tweets, neg_tweets], 
                                hole = 0.6
                                )
                            ]
                    )

    fig.update_traces(
                hoverinfo = 'label', 
                marker = dict(colors = ['#3eb869', '#d6993c', '#e35970']), 
                textposition = 'inside'
                )

    fig.update_layout(
                template = 'custom', 
                showlegend = False, 
                margin = dict(t=0, b=0, l=0, r=0), 
                width = fig_width_1-20, 
                height = fig_height-20
                )

    return fig

# Stacked area graph of positive, neutral, negative sentiment over time
def stacked_area(df):
    # Calculating sentiment by day
    sentiment_count = {'positive': [], 'neutral': [], 'negative': []}

    # Iterating over every date
    for date in df['date_short'].unique():
        dff = df.loc[df['date_short'] == date]
        
        # Counting the number of positive, neutral, and negative tweets and adding them to the dictionary
        sentiment_count['positive'].append(dff.loc[dff['compound'] > 0].count()[0])
        sentiment_count['neutral'].append(dff.loc[dff['compound'] == 0].count()[0])
        sentiment_count['negative'].append(dff.loc[dff['compound'] < 0].count()[0])

    dates = df['date_short'].unique()

    # Stacked area graph code credit: https://plotly.com/python/filled-area-plots/
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x = dates, 
        y = sentiment_count['neutral'],
        mode = 'lines',
        line = dict(width = 0.5, color = '#d6993c'),
        fillcolor = 'rgba(214, 153, 60, 0.75)',
        stackgroup = 'one',
        groupnorm = 'percent',
        name = 'neutral',
        hoverinfo = 'name+y'
    ))

    fig.add_trace(go.Scatter(
        x = dates, 
        y = sentiment_count['negative'],
        mode = 'lines',
        line = dict(width = 0.5, color = '#e35970'),
        fillcolor = 'rgba(227, 89, 112, 0.75)',
        stackgroup = 'one',
        name = 'negative',
        hoverinfo = 'name+y'
    ))

    fig.add_trace(go.Scatter(
        x = dates, 
        y = sentiment_count['positive'],
        mode = 'lines',
        line = dict(width = 0.5, color = '#3eb869'),
        fillcolor = 'rgba(62, 184, 105, 0.75)',
        stackgroup = 'one',
        name = 'positive',
        hoverinfo = 'name+y'
    ))

    fig.update_layout(
        xaxis = dict(showgrid = False),
        yaxis = dict(
            type = 'linear',
            range = [1,100],
            ticksuffix = '%'
        ),
        hovermode = 'x unified',
        template = 'custom',
        showlegend = False,
        margin = dict(t=20, l=45, b=25, r=0),
        height = fig_height-20,
    )

    return fig

# Gauge plot
def gauge(df):
    compound = df.loc[df['compound'] != 0]['compound'].mean()

    fig = go.Figure(go.Indicator(
    mode = 'gauge+number',
    value = compound,
    domain = {'x': [0, 1], 'y': [0, 1]},
    gauge = {
            'axis': {'range': [-1, 1]},
            'steps': 
            [
                {'range': [-1, -0.05], 'color': color_red},
                {'range': [-0.05, 0.05], 'color': color_orange},
                {'range': [0.05, 1], 'color': color_green}
            ],
            'bar': {'color': 'white'}
            }
    ))

    fig.update_layout(
            template = 'custom',
            height = fig_height-20,
            width = fig_width_1-20,
            margin = dict(t=0, b=0, l=5, r=5)
            )

    return fig

# Figure showing number of tweets over time. Fourier transform will be applied for interactive smoothing of the line graph
def line_count(tweet_count):
    data = [x['tweet_count'] for x in tweet_count.data][0:-1]
    dates = [x['end'] for x in tweet_count.data][0:-1]

    fig = go.Figure(data = go.Scatter(x = dates, y = data, line = dict(color = '#f0f0f0')))

    fig.update_layout(
            yaxis_title = 'Number of Tweets', 
            template = 'custom', 
            hovermode = 'x unified',
            width = fig_width_2-20, 
            height = fig_height-20,
            margin = dict(t=10, b=25, l=60, r=20),
            xaxis_tickformat='%b %d'
            )

    # Annotating the maximum and minimum values.
    fig.add_annotation(
            showarrow = False,
            yshift = 10,
            # Code credit for finding index position of maximum value: user: 'too much php', https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
            x = dates[data.index(max(data))],
            y = max(data),
            text = 'Max: ' + str(max(data))
            )

    fig.add_annotation(
            showarrow = False,
            yshift = -10,
            # Code credit for finding index position of maximum value: user: 'too much php', https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
            x = dates[data.index(min(data))],
            y = min(data),
            text = 'Min: ' + str(min(data))
            )                

    return fig

# Sentiment line graph
def sentiment_line(df):
    sentiment_count = {'positive': [], 'neutral': [], 'negative': []}

    # Iterating over every date
    for date in df.groupby('date_short').mean().index:
        dff = df.loc[df['date_short'] == date]
        
        # Calculating the percent of positive, neutral, and negative tweets and adding them to the dictionary
        sentiment_count['positive'].append((dff.loc[dff['compound'] > 0].count()[0] / dff.shape[0])*100)
        sentiment_count['neutral'].append((dff.loc[dff['compound'] == 0].count()[0] / dff.shape[0])*100)
        sentiment_count['negative'].append((dff.loc[dff['compound'] < 0].count()[0] / dff.shape[0])*100)

    x = df.groupby('date_short').mean().index

    fig = go.Figure(data=go.Scatter(
        x = x,
        y = sentiment_count['positive'],
        name = 'Positive',
        line = dict(color = color_green)
    ))

    fig.add_trace(go.Scatter(
        x = x,
        y = sentiment_count['neutral'],
        name = 'Neutral',
        line = dict(color = color_orange)
    ))

    fig.add_trace(go.Scatter(
        x = x,
        y = sentiment_count['negative'],
        name = 'Negative',
        line = dict(color = color_red)
    ))

    fig.update_layout(
            template = 'custom',
            xaxis_tickformat='%b %d',
            xaxis_range = [df.groupby('date_short').mean().index[0], df.groupby('date_short').mean().index[-1]],
            hovermode='x unified',
            showlegend = False,
            xaxis=dict(showgrid=False),
            yaxis=dict(ticksuffix='%'),
            margin = dict(t=20, l=45, b=25, r=0),
            height = fig_height-20
            )

    return fig

# Word cloud
def pos_word_cloud(df):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
                    font_path = './open-sans/OpenSans-Semibold.ttf', 
                    stopwords = stopwords, 
                    background_color = bg_color, 
                    colormap = 'Blues',
                    width = fig_width_2, 
                    height = fig_height).generate(' '.join(df.loc[df['compound'] > 0]['clean_tweet'])
                    )

    return wordcloud.to_image()

def neg_word_cloud(df):
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
                    font_path = './open-sans/OpenSans-Semibold.ttf', 
                    stopwords = stopwords, 
                    background_color = bg_color, 
                    colormap = 'Reds',
                    width = fig_width_2, 
                    height = fig_height).generate(' '.join(df.loc[df['compound'] < 0]['clean_tweet'])
                    )

    return wordcloud.to_image()


# *********************************** DASHBOARD HTML AND CALLBACKS ***********************************

# Importing free icons from FontAwesome. The license allows use on most projects: https://fontawesome.com/license/free
font_awesome = 'https://use.fontawesome.com/releases/v5.10.2/css/all.css'

meta_tags = [{"name": "viewport", "content": "width=device-width"}]
external_stylesheets = [meta_tags, font_awesome]

app = dash.Dash(__name__, external_stylesheets = external_stylesheets, suppress_callback_exceptions = False)
app.title = 'Twitter Sentiment Analysis Dashboard'

# HTML for the dashboard homepage
app.layout = html.Div(
    id='root',
    children=[
        html.Div(id='home', children=[
            html.Div(className='start-header', children=[
                html.H1('Twitter Sentiment Analysis Dashboard'),
                html.Div(className='search-bar-container', children=[
                    dcc.Input(id='search-start', value='', placeholder='Enter a hashtag...', debounce=True, type='text', n_submit=0, autoComplete='off', className='start-search-bar', autoFocus=True),
                    html.I(className='fas fa-search')
                ])
            ]),
            html.P('Select Classification Model', style={'text-align': 'center', 'margin-top': '25px', 'opacity': '60%'}),
            html.Div(className='home-model-dropdown', children=[
                html.Div(className='model-dropdown', children=[
                    dcc.Dropdown([{
                        'label': html.Div(['VADER'], style={'color': 'white'}),
                        'value': 'VADER'
                    }, 
                    {
                        'label': html.Div(['TextBlob'], style={'color': 'white'}),
                        'value': 'TextBlob'
                    }], value = 'VADER', id='home-model-select', clearable=False, searchable=False, style={'background-color': bg_color,
                                                                                                           'color': 'white',
                                                                                                           'width': '120px',
                                                                                                           'border': 'none',
                                                                                                           'outline': 'none'})
                ])
            ])
        ]),
    ]
)


# Callback for updating the page layout when the first search has been entered
@app.callback(
    Output('root', 'children'),
    Input('search-start', 'n_submit'),
    State('search-start', 'value'),
    State('home-model-select', 'value'),
    # Prevents the callback being triggered instantly
    prevent_initial_call=True
)
def update_html(n_submit, value, model):
    # If the search doesn't contain a hashtag at the start: add a hashtag
    if not value.startswith('#'):
        value = '#' + value

    # Setting date to current time minus 1 minute
    date = datetime.now() - timedelta(minutes=1)
    date = date.astimezone(pytz.utc)

    # Querying Twitter API to retrieve tweets and store them in a dataframe with their sentiment classification
    df = retrieve_tweets(value, num_days, num_tweets, model)

    # Converting dataframe to dictionary format to be stored in the app to be downloaded by the user later
    df_dict = df.to_dict()

    # Making filename for data download
    # format: hashtag-data-YYYY-MM-DD
    filename = value + '-data-' + datetime.now().date().strftime('%Y-%m-%d') + '.csv'
    filename = filename.strip('#')

    # Percentage of positive, neutral and negative tweets
    pos_pct, neu_pct, neg_pct = sentiment_percent(df)

    # Calculating compound score
    compound_score = round(df.loc[df['compound'] != 0]['compound'].mean(), 3)

    # Getting number of tweets containing the queried term
    tweet_count = number_tweets(value)

    # Getting the embed code for positive and negative sample tweets
    pos_embed_code = get_embed(df, 'positive')
    neg_embed_code = get_embed(df, 'negative')

    # Generating the figures
    fig_pie1 = pie_chart1(df)
    fig_pie2 = pie_chart2(df)
    fig_stacked = stacked_area(df)
    fig_line_count = line_count(tweet_count)
    fig_gauge = gauge(df)
    fig_sentiment_line = sentiment_line(df)

    # Creating positive and negative word clouds
    # Dash word cloud code credit: user: 'Randy' https://stackoverflow.com/questions/58907867/how-to-show-wordcloud-image-on-dash-web-application
    pos_img = BytesIO()
    pos_word_cloud(df).save(pos_img, format='PNG')
    pos_word_cloud_img = 'data:image/png;base64,{}'.format(base64.b64encode(pos_img.getvalue()).decode())

    neg_img = BytesIO()
    neg_word_cloud(df).save(neg_img, format='PNG')
    neg_word_cloud_img = 'data:image/png;base64,{}'.format(base64.b64encode(neg_img.getvalue()).decode())

    # Returning HTML for the page which contains all figures and statistics
    return (
    # --- HEADER CONTAINING: TITLE, SEARCH BAR, MODEL SELECTION, DATA DOWNLOAD --- #
    html.Div(id='header', className='search-header', children=[
        html.Div(className='container-fluid', children=[
            html.Div(className='row', style={'text-align': 'center', 'padding-top': '20px'}, children=[
                html.Div(className='col-md', style={'text-align': 'left', 'margin-top': '-3px'}, children=[
                    html.H3('Twitter Sentiment Analysis Dashboard')
                ]),
                # --- SEARCH BAR --- #
                html.Div(className='col-md-3', children=[
                    html.Div(className='search-bar-container', children=[
                        dcc.Input(id='search', className='search-bar', value='', placeholder='Enter a hashtag...', debounce=True, type='text', n_submit=0, autoComplete='off', autoFocus=True),
                        html.I(className='fas fa-search')
                    ])
                ]),
                html.Div(className='col-md download-btn', children=[
                    # --- MODEL SELECTION DROPDOWN --- #
                    html.Div(className='col-md-6', children=[
                        html.Div(className='model-dropdown', children=[
                            dcc.Dropdown([{
                                'label': html.Div(['VADER'], style={'color': 'white'}),
                                'value': 'VADER'
                            }, 
                            {
                                'label': html.Div(['TextBlob'], style={'color': 'white'}),
                                'value': 'TextBlob'
                            }], value = model, id='model-select', clearable=False, searchable=False, style={'background-color': bg_color,
                                                                                                            'color': 'white',
                                                                                                            'width': '120px',
                                                                                                            'border': 'none',
                                                                                                            'outline': 'none'})
                        ])
                    ]),
                    # --- DATA DOWNLOAD BUTTON --- #
                    html.Div(className='col-md-6', children=[
                        html.Div(className='download-container', children=[
                            html.I(className='fas fa-download'),
                            html.Button('Download data', id='data-download-btn', className='download-btn-text', n_clicks=0),
                            dcc.Download(id='data-download')
                        ])
                    ])
                ])
            ])
        ])
    ]),
    # --- ENTERED HASHTAG --- #
    html.Div(id='entered-hashtag-text', className='entered-hashtag', children=[html.H2(value)]),
    # --- FIRST ROW OF STATISTICS --- #
    html.Div(className='stat-wrapper', children=[
        html.Div(className='top-row', children=[
            html.Div(className='row', style={'margin-bottom': '2rem'}, children=[
                # --- POSITIVE STATISTIC --- #
                html.Div(className='col-md-3', children=[
                    html.Div(className='card', children=[
                        html.Div(className='card-body', children=[
                            html.Div(className='stat-container', style={'color': color_green}, children=[
                                html.Div(className='stat-left', children=[
                                    html.P('Positive'),
                                    html.P(id='positive-stat', style={'font-size': '2em', 'margin-top': '-25px'}, children=str(pos_pct)+'%')
                                ]),
                                html.Div(className='stat-right', children=[
                                    html.I(className='fas fa-smile')
                                ])
                            ])
                        ])
                    ])
                ]),
                # --- NEUTRAL STATISTIC --- #
                html.Div(className='col-md-3', children=[
                    html.Div(className='card', children=[
                        html.Div(className='card-body', children=[
                            html.Div(className='stat-container', style={'color': color_orange}, children=[
                                html.Div(className='stat-left', children=[
                                    html.P('Neutral'),
                                    html.P(id='neutral-stat', style={'font-size': '2em', 'margin-top': '-25px'}, children=str(neu_pct)+'%')
                                ]),
                                html.Div(className='stat-right', children=[
                                    html.I(className='fas fa-meh')
                                ])
                            ])
                        ])
                    ])
                ]),
                # --- NEGATIVE STATISTIC --- #
                html.Div(className='col-md-3', children=[
                    html.Div(className='card', children=[
                        html.Div(className='card-body', children=[
                            html.Div(className='stat-container', style={'color': color_red}, children=[
                                html.Div(className='stat-left', children=[
                                    html.P('Negative'),
                                    html.P(id='negative-stat', style={'font-size': '2em', 'margin-top': '-25px'}, children=str(neg_pct)+'%')
                                ]),
                                html.Div(className='stat-right', children=[
                                    html.I(className='fas fa-frown')
                                ])
                            ])
                        ])
                    ])
                ]),
                # --- COMPOUND STATISTIC --- #
                html.Div(className='col-md-3', children=[
                    html.Div(className='card', children=[
                        html.Div(className='card-body', children=[
                            html.Div(className='compound-stat', children=[
                                # Tooltip code credit: https://www.w3schools.com/css/css_tooltip.asp
                                html.Div(className='compound-tooltip', children=[
                                    html.I(className='fas fa-info-circle'),
                                    html.Span(className='compound-tooltip-text', children='A value from -1 (absolute negative sentiment) to +1 (absolute positive sentiment)')
                                ]),
                                html.P('Overall Sentiment'),
                                html.P(id='compound-stat', style={'font-size': '2em', 'margin-top': '-25px'}, children=compound_score)
                            ])
                        ])
                    ])
                ])
            ])
        ])
    ]),
    # --- CONTAINER FOR ALL GRAPHS --- #
    html.Div(className='row-container', children=[
        # --- FIRST ROW OF GRAPHS --- #
        html.Div(className='row', children=[
            # --- PIE CHART --- #
            html.Div(className='col-md', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Pie chart showing the percentage of positive, neutral and negative tweets')
                ]),
                html.Div(className='card', style={'width': str(fig_width_1)+'px'}, children=[
                    html.Div(className='card-header', children='Sentiment Pie Chart'),
                    html.Div(className='card-body', children=[
                        dcc.Graph(id='pie_chart2', figure=fig_pie2)
                    ])
                ])
            ]),
            # --- STACKED AREA GRAPH --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Stacked area graph of sentiment over time')
                ]),
                html.Div(className='card', style={}, children=[
                    html.Div(className='card-header', children='Sentiment Stacked Area Graph'),
                    html.Div(className='card-body', children=[
                        dcc.Graph(id='stacked_area', figure=fig_stacked)
                    ])
                ])
            ]),
            # --- GAUGE CHART --- #
            html.Div(className='col-md', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Gauge chart of compound score (overall sentiment)')
                ]),
                html.Div(className='card', style={'width': str(fig_width_1)+'px'}, children=[
                    html.Div(className='card-header', children='Overall Sentiment'),
                    html.Div(className='card-body', children=[
                        dcc.Graph(id='gauge', figure=fig_gauge)
                    ])
                ])
            ])
        ]),
        # --- SECOND ROW OF GRAPHS --- #
        html.Div(className='row', children=[
            # --- SENTIMENT LINE GRAPH --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Positive, neutral and negative percentages over time')
                ]),
                html.Div(className='card', children=[
                    html.Div(className='card-header', children='Sentiment Line Graph'),
                    html.Div(className='card-body', children=[
                        dcc.Graph(id='sentiment_line', figure=fig_sentiment_line)
                    ])
                ])
            ]),
            # --- NUMBER OF TWEETS GRAPH --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Number of tweets containing the entered hashtag over the past week')
                ]),
                html.Div(className='card', children=[
                    html.Div(className='card-header', children='Number of Tweets'),
                    html.Div(className='card-body', children=[
                        dcc.Graph(id='line_count', figure=fig_line_count)
                    ])
                ])
            ]),
        ]),
        # --- THIRD ROW OF GRAPHS --- #
        html.Div(className='row', children=[
            # --- POSITIVE WORD CLOUD --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Word cloud containing words from positive tweets')
                ]),
                html.Div(className='card', children=[
                    html.Div(className='card-header', children='Positive Word Cloud'),
                    html.Div(className='card-body', children=[
                        html.Img(id='pos_word_cloud', src=pos_word_cloud_img)
                    ])
                ])
            ]),
            # --- NEGATIVE WORD CLOUD --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='Word cloud containing words from negative tweets')
                ]),
                html.Div(className='card', children=[
                    html.Div(className='card-header', children='Negative Word Cloud'),
                    html.Div(className='card-body', children=[
                        html.Img(id='neg_word_cloud', src=neg_word_cloud_img)
                    ])
                ])
            ]),
        ]),
        # --- BOTTOM ROW OF GRAPHS --- #
        html.Div(className='row', children=[
            # --- SAMPLE POSITIVE TWEET --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='A sample tweet which has been classified as positive')
                ]),
                html.Div(className='card', children=[
                    html.Div(className='card-header', children='Sample Positive Tweet'),
                    html.Div(className='card-body', children=[
                        html.Iframe(id='pos_tweet', srcDoc=pos_embed_code, width='100%', height=300)
                    ])
                ])
            ]),
            # --- SAMPLE NEGATIVE TWEET --- #
            html.Div(className='col-md-6', children=[
                html.Div(className='card-tooltip', children=[
                    html.I(className='fas fa-info-circle'),
                    html.Span(className='card-tooltip-text', children='A sample tweet which has been classified as negative')
                ]),
                html.Div(className='card', children=[
                    html.Div(className='card-header', children='Sample Negative Tweet'),
                    html.Div(className='card-body', children=[
                        html.Iframe(id='neg_tweet', srcDoc=neg_embed_code, width='100%', height=300)
                    ])
                ])
            ])
        ])
    ]),
    # Storing data to be accessed in future callbacks
    dcc.Store(id='stored-data', data = df_dict),
    dcc.Store(id='entered-hashtag', data = filename)
    )


# Callback for updating the page when a new search has been entered
@app.callback(
    Output('search', 'value'),
    Output('entered-hashtag-text', 'children'),
    Output('positive-stat', 'children'),
    Output('neutral-stat', 'children'),
    Output('negative-stat', 'children'),
    Output('compound-stat', 'children'),
    Output('pie_chart2', 'figure'),
    Output('stacked_area', 'figure'),
    Output('gauge', 'figure'),
    Output('sentiment_line', 'figure'),
    Output('line_count', 'figure'),
    Output('pos_word_cloud', 'src'),
    Output('neg_word_cloud', 'src'),
    Output('pos_tweet', 'srcDoc'),
    Output('neg_tweet', 'srcDoc'),
    Output('stored-data', 'data'),
    Output('entered-hashtag', 'data'),
    Input('search', 'value'),
    Input('model-select', 'value')
)
def update_figures(value, model):
    # If the search doesn't contain a hashtag at the start: add a hashtag
    if not value.startswith('#'):
        value = '#' + value

    # Setting date to current time minus 1 minute
    date = datetime.now() - timedelta(minutes=1)
    date = date.astimezone(pytz.utc)

    # Querying Twitter API to retrieve tweets and store them in a dataframe with their sentiment classification
    df = retrieve_tweets(value, num_days, num_tweets, model)

    # Converting dataframe to dictionary format to be stored in the app to be downloaded by the user later
    df_dict = df.to_dict()

    # Making filename for data download
    # format: hashtag-data-YYYY-MM-DD
    filename = value + '-data-' + datetime.now().date().strftime('%Y-%m-%d') + '.csv'
    filename = filename.strip('#')

    # Percentage of positive, neutral and negative tweets
    pos_pct, neu_pct, neg_pct = sentiment_percent(df)

    compound_score = round(df.loc[df['compound'] != 0]['compound'].mean(), 3)

    # Getting number of tweets containing the queried term
    tweet_count = number_tweets(value)

    # Getting the embed code for positive and negative sample tweets
    pos_embed_code = get_embed(df, 'positive')
    neg_embed_code = get_embed(df, 'negative')

    # Generating the figures
    fig_pie1 = pie_chart1(df)
    fig_pie2 = pie_chart2(df)
    fig_stacked = stacked_area(df)
    fig_line_count = line_count(tweet_count)
    fig_gauge = gauge(df)
    fig_sentiment_line = sentiment_line(df)

    # Creating positive and negative word clouds
    # Dash word cloud code credit: user: 'Randy' https://stackoverflow.com/questions/58907867/how-to-show-wordcloud-image-on-dash-web-application
    pos_img = BytesIO()
    pos_word_cloud(df).save(pos_img, format='PNG')
    pos_word_cloud_img = 'data:image/png;base64,{}'.format(base64.b64encode(pos_img.getvalue()).decode())

    neg_img = BytesIO()
    neg_word_cloud(df).save(neg_img, format='PNG')
    neg_word_cloud_img = 'data:image/png;base64,{}'.format(base64.b64encode(neg_img.getvalue()).decode())

    # Reset the search bar text to empty
    search_text = ''

    return (
            search_text, 
            html.H2(value), 
            str(pos_pct)+'%', 
            str(neu_pct)+'%', 
            str(neg_pct)+'%', 
            compound_score, 
            fig_pie2, 
            fig_stacked, 
            fig_gauge, 
            fig_sentiment_line, 
            fig_line_count, 
            pos_word_cloud_img, 
            neg_word_cloud_img, 
            pos_embed_code,
            neg_embed_code,
            df_dict, 
            filename,
            )


# Callback for downloading the dataset
@app.callback(
    Output('data-download', 'data'),
    Output('data-download-btn', 'n_clicks'),
    Input('data-download-btn', 'n_clicks'),
    Input('stored-data', 'data'),
    Input('entered-hashtag', 'data'),
    prevent_initial_call = True
)
def download_data(n_clicks, data, filename):
    # Checking if button has been clicked -- this is a workaround for a bug where the download is triggered when the html is changed (i.e a new search is entered)
    if n_clicks != 0:
        # Loading in the saved dataframe
        df = pd.DataFrame.from_dict(data)

        return dcc.send_data_frame(df.to_csv, filename), 0


if __name__ == "__main__":
    app.run_server(debug = True, dev_tools_ui = False, dev_tools_props_check = False)
    # app.run_server(debug=True)