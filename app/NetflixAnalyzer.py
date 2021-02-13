import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine
import pymysql
import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from collections import Counter
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

# Retrieve data from github repo --------------------------------------------------------------
@st.cache
def get_data():
    url = 'https://raw.githubusercontent.com/konstantinos-giannopoulos/MovieAnalysis/main/netflix_titles.csv'
    df = pd.read_csv(url)
    return df

df = get_data()

# Interaction with database -------------------------------------------------------------------
tableName = 'netflix_titles'
sqlEngine = create_engine('mysql+pymysql://root:pwd@mysqldb:3306/netflix_db')
dbConnection = sqlEngine.connect()
try:
    df.to_sql(tableName, dbConnection, if_exists='fail')
except:
    pass
finally:
    netflix_data = pd.read_sql("select * from netflix_titles", dbConnection)
    dbConnection.close()

# Preprocess data ------------------------------------------------------------------------------
netflix_data["date_added"] = pd.to_datetime(netflix_data['date_added'])
netflix_data['year_added'] = netflix_data['date_added'].dt.year
netflix_data['month_added'] = netflix_data['date_added'].dt.month

netflix_data['season_count'] = netflix_data.apply(lambda x : x['duration'].split(" ")[0] if 'Season' in x['duration'] else '', axis = 1)
netflix_data['duration'] = netflix_data.apply(lambda x : x['duration'].split(" ")[0] if 'Season' not in x['duration'] else '', axis = 1)

# Sidebar -------------------------------------------------------------------------------------------
selected = st.sidebar.selectbox('Please choose: ', ['Dashboards and Analytics', 'Recommendations', 'Informations'])
with st.sidebar.beta_expander('ABOUT'):
    st.write('A data analysis application that developed for my M.Sc. in Big Data and Analytics at University of Piraeus.')
    st.write('Technologies used: Python, MySQL, Docker')

if selected == 'Dashboards and Analytics':

    st.title('Dashboards and Analytics')

    # Movies vs TV Shows bar plot --------------------------------------------------------------------
    if st.checkbox('Movies vs TV shows'):
        st.subheader('Movies vs TV shows')
        sns.set(style="darkgrid")
        fig = plt.figure()
        ax = sns.countplot(x="type", data=netflix_data, palette="Set2")
        st.pyplot(fig)

    # Movies vs TV Shows pie chart ----------------------------------------------------------------------------------
        group_netflix = netflix_data.type.value_counts()
        trace = go.Pie(labels=group_netflix.index,values=group_netflix.values,pull=[0.05])
        layout = go.Layout(title='Movies vs TV shows pie chart', height=400, legend=dict(x=1.1, y=1.3))
        fig = go.Figure(data=[trace],layout=layout)
        fig.update_layout(height=500,width=700)
        st.plotly_chart(fig)

    # Growth by year plots -------------------------------------------------------------------------------------
    netflix_tv = netflix_data[netflix_data["type"] == "TV Show"]
    netflix_movies = netflix_data[netflix_data["type"] == "Movie"]    
    
    if st.checkbox('Growth by year'):
        st.subheader('Content added over years')

        col = "year_added"

        vc1 = netflix_tv[col].value_counts().reset_index()
        vc1 = vc1.rename(columns = {col : "count", "index" : col})
        vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
        vc1 = vc1.sort_values(col)

        vc2 = netflix_movies[col].value_counts().reset_index()
        vc2 = vc2.rename(columns = {col : "count", "index" : col})
        vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
        vc2 = vc2.sort_values(col)

        trace1 = go.Scatter(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
        trace2 = go.Scatter(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
        data = [trace1, trace2]
        layout = go.Layout(legend=dict(x=0.1, y=1.1, orientation="h"))
        fig = go.Figure(data, layout=layout)
        fig.update_xaxes(title_text='Years')
        fig.update_yaxes(title_text='Number of content added')
        st.plotly_chart(fig)

        trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
        trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
        data = [trace1, trace2]
        layout = go.Layout(legend=dict(x=0.1, y=1.1, orientation="h"))     
        fig = go.Figure(data, layout=layout)
        fig.update_xaxes(title_text='Years')
        fig.update_yaxes(title_text='Number of content added')
        st.plotly_chart(fig)

    # Growth by month plots -------------------------------------------------------------------------------
    if st.checkbox('Added by month'):    
        st.subheader('Content added by month')

        col = 'month_added'

        vc1 = netflix_tv[col].value_counts().reset_index()
        vc1 = vc1.rename(columns = {col : "count", "index" : col})
        vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
        vc1 = vc1.sort_values(col)

        trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
        data = [trace1]
        layout = go.Layout(legend=dict(x=0.1, y=1.1, orientation="h"))
        fig = go.Figure(data, layout=layout)
        fig.update_xaxes(title_text='Month')
        fig.update_yaxes(title_text='Number of content added')       
        st.plotly_chart(fig)

    # Duration distribution ---------------------------------------------------------------------------------------------------
    if st.checkbox('Duration'):
        st.subheader('Duration distribution')

        x1 = netflix_movies['duration'].fillna(0.0).astype(float)
        fig = ff.create_distplot([x1], [''], bin_size=0.7, curve_type='normal', colors=["#6ad49b"])
        st.plotly_chart(fig)

    # The 15 oldest movies and tv series on Netflix ---------------------------------------------------------------------------
    if st.checkbox('Oldest movies and TV series on Netflix'):
        st.subheader("The 15 oldest movies on Netflix")
        small = netflix_data.sort_values("release_year", ascending = True)
        small = small[small['duration'] != ""]
        st.write(small[['title', "release_year"]][:15])

        st.subheader("The 15 oldest tv series on Netflix")
        small = netflix_data.sort_values("release_year", ascending = True)
        small = small[small['season_count'] != ""]
        st.write(small[['title', "release_year"]][:15])

    # Most productive countries ------------------------------------------------------------------------------------------------
    if st.checkbox('Most productive countries'):
        st.subheader('Content by country')
        country_codes = {'afghanistan': 'AFG',
        'albania': 'ALB',
        'algeria': 'DZA',
        'american samoa': 'ASM',
        'andorra': 'AND',
        'angola': 'AGO',
        'anguilla': 'AIA',
        'antigua and barbuda': 'ATG',
        'argentina': 'ARG',
        'armenia': 'ARM',
        'aruba': 'ABW',
        'australia': 'AUS',
        'austria': 'AUT',
        'azerbaijan': 'AZE',
        'bahamas': 'BHM',
        'bahrain': 'BHR',
        'bangladesh': 'BGD',
        'barbados': 'BRB',
        'belarus': 'BLR',
        'belgium': 'BEL',
        'belize': 'BLZ',
        'benin': 'BEN',
        'bermuda': 'BMU',
        'bhutan': 'BTN',
        'bolivia': 'BOL',
        'bosnia and herzegovina': 'BIH',
        'botswana': 'BWA',
        'brazil': 'BRA',
        'british virgin islands': 'VGB',
        'brunei': 'BRN',
        'bulgaria': 'BGR',
        'burkina faso': 'BFA',
        'burma': 'MMR',
        'burundi': 'BDI',
        'cabo verde': 'CPV',
        'cambodia': 'KHM',
        'cameroon': 'CMR',
        'canada': 'CAN',
        'cayman islands': 'CYM',
        'central african republic': 'CAF',
        'chad': 'TCD',
        'chile': 'CHL',
        'china': 'CHN',
        'colombia': 'COL',
        'comoros': 'COM',
        'congo democratic': 'COD',
        'Congo republic': 'COG',
        'cook islands': 'COK',
        'costa rica': 'CRI',
        "cote d'ivoire": 'CIV',
        'croatia': 'HRV',
        'cuba': 'CUB',
        'curacao': 'CUW',
        'cyprus': 'CYP',
        'czech republic': 'CZE',
        'denmark': 'DNK',
        'djibouti': 'DJI',
        'dominica': 'DMA',
        'dominican republic': 'DOM',
        'ecuador': 'ECU',
        'egypt': 'EGY',
        'el salvador': 'SLV',
        'equatorial guinea': 'GNQ',
        'eritrea': 'ERI',
        'estonia': 'EST',
        'ethiopia': 'ETH',
        'falkland islands': 'FLK',
        'faroe islands': 'FRO',
        'fiji': 'FJI',
        'finland': 'FIN',
        'france': 'FRA',
        'french polynesia': 'PYF',
        'gabon': 'GAB',
        'gambia, the': 'GMB',
        'georgia': 'GEO',
        'germany': 'DEU',
        'ghana': 'GHA',
        'gibraltar': 'GIB',
        'greece': 'GRC',
        'greenland': 'GRL',
        'grenada': 'GRD',
        'guam': 'GUM',
        'guatemala': 'GTM',
        'guernsey': 'GGY',
        'guinea-bissau': 'GNB',
        'guinea': 'GIN',
        'guyana': 'GUY',
        'haiti': 'HTI',
        'honduras': 'HND',
        'hong kong': 'HKG',
        'hungary': 'HUN',
        'iceland': 'ISL',
        'india': 'IND',
        'indonesia': 'IDN',
        'iran': 'IRN',
        'iraq': 'IRQ',
        'ireland': 'IRL',
        'isle of man': 'IMN',
        'israel': 'ISR',
        'italy': 'ITA',
        'jamaica': 'JAM',
        'japan': 'JPN',
        'jersey': 'JEY',
        'jordan': 'JOR',
        'kazakhstan': 'KAZ',
        'kenya': 'KEN',
        'kiribati': 'KIR',
        'north korea': 'PRK',
        'south korea': 'KOR',
        'kosovo': 'KSV',
        'kuwait': 'KWT',
        'kyrgyzstan': 'KGZ',
        'laos': 'LAO',
        'latvia': 'LVA',
        'lebanon': 'LBN',
        'lesotho': 'LSO',
        'liberia': 'LBR',
        'libya': 'LBY',
        'liechtenstein': 'LIE',
        'lithuania': 'LTU',
        'luxembourg': 'LUX',
        'macau': 'MAC',
        'macedonia': 'MKD',
        'madagascar': 'MDG',
        'malawi': 'MWI',
        'malaysia': 'MYS',
        'maldives': 'MDV',
        'mali': 'MLI',
        'malta': 'MLT',
        'marshall islands': 'MHL',
        'mauritania': 'MRT',
        'mauritius': 'MUS',
        'mexico': 'MEX',
        'micronesia': 'FSM',
        'moldova': 'MDA',
        'monaco': 'MCO',
        'mongolia': 'MNG',
        'montenegro': 'MNE',
        'morocco': 'MAR',
        'mozambique': 'MOZ',
        'namibia': 'NAM',
        'nepal': 'NPL',
        'netherlands': 'NLD',
        'new caledonia': 'NCL',
        'new zealand': 'NZL',
        'nicaragua': 'NIC',
        'nigeria': 'NGA',
        'niger': 'NER',
        'niue': 'NIU',
        'northern mariana islands': 'MNP',
        'norway': 'NOR',
        'oman': 'OMN',
        'pakistan': 'PAK',
        'palau': 'PLW',
        'panama': 'PAN',
        'papua new guinea': 'PNG',
        'paraguay': 'PRY',
        'peru': 'PER',
        'philippines': 'PHL',
        'poland': 'POL',
        'portugal': 'PRT',
        'puerto rico': 'PRI',
        'qatar': 'QAT',
        'romania': 'ROU',
        'russia': 'RUS',
        'rwanda': 'RWA',
        'saint kitts and nevis': 'KNA',
        'saint lucia': 'LCA',
        'saint martin': 'MAF',
        'saint pierre and miquelon': 'SPM',
        'saint vincent and the grenadines': 'VCT',
        'samoa': 'WSM',
        'san marino': 'SMR',
        'sao tome and principe': 'STP',
        'saudi arabia': 'SAU',
        'senegal': 'SEN',
        'serbia': 'SRB',
        'seychelles': 'SYC',
        'sierra leone': 'SLE',
        'singapore': 'SGP',
        'sint maarten': 'SXM',
        'slovakia': 'SVK',
        'slovenia': 'SVN',
        'solomon islands': 'SLB',
        'somalia': 'SOM',
        'south africa': 'ZAF',
        'south sudan': 'SSD',
        'spain': 'ESP',
        'sri lanka': 'LKA',
        'sudan': 'SDN',
        'suriname': 'SUR',
        'swaziland': 'SWZ',
        'sweden': 'SWE',
        'switzerland': 'CHE',
        'syria': 'SYR',
        'taiwan': 'TWN',
        'tajikistan': 'TJK',
        'tanzania': 'TZA',
        'thailand': 'THA',
        'timor-leste': 'TLS',
        'togo': 'TGO',
        'tonga': 'TON',
        'trinidad and tobago': 'TTO',
        'tunisia': 'TUN',
        'turkey': 'TUR',
        'turkmenistan': 'TKM',
        'tuvalu': 'TUV',
        'uganda': 'UGA',
        'ukraine': 'UKR',
        'united arab emirates': 'ARE',
        'united kingdom': 'GBR',
        'united states': 'USA',
        'uruguay': 'URY',
        'uzbekistan': 'UZB',
        'vanuatu': 'VUT',
        'venezuela': 'VEN',
        'vietnam': 'VNM',
        'virgin islands': 'VGB',
        'west bank': 'WBG',
        'yemen': 'YEM',
        'zambia': 'ZMB',
        'zimbabwe': 'ZWE'}
    
        colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
            "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
            "#08519c", "#0b4083", "#08306b"
        ]
            
        def get_country(ddf):
            country_with_code, country = {}, {}
            shows_countries = ", ".join(ddf['country'].dropna()).split(", ")
            for c,v in dict(Counter(shows_countries)).items():
                code = ""
                if c.lower() in country_codes:
                    code = country_codes[c.lower()]
                country_with_code[code] = v
                country[c] = v

            return country

        country_vals = get_country(netflix_data)
        tabs = Counter(country_vals).most_common(25)

        labels = [_[0] for _ in tabs][::-1]
        values = [_[1] for _ in tabs][::-1]
        trace1 = go.Bar(y=labels, x=values, orientation="h", name="", marker=dict(color="#a678de"))

        data = [trace1]
        layout = go.Layout(title="Countries with most content", height=700, legend=dict(x=0.1, y=1.1, orientation="h"))
        fig = go.Figure(data, layout=layout)
        fig = fig.update_xaxes(title_text="Number of content")
        st.plotly_chart(fig)

    # Movies / TV shows by country for each year (map) -------------------------------------------------------------------------
    if st.checkbox('Releases by country'):
        df_country_year = netflix_data.groupby(by=['country', 'type', 'year_added']).count().reset_index()
        df_country_year['aggregate'] = df_country_year.groupby(by=['country'])['title'].cumsum()

        fig = px.choropleth(df_country_year.sort_values(by='year_added'), 
                            locations='country', color='aggregate', 
                            locationmode='country names', animation_frame='year_added', 
                            range_color=[0, 500],)
        st.plotly_chart(fig)

    # TV shows and seasons -----------------------------------------------------------------------------------------------------
    if st.checkbox('Seasons'):
        st.subheader('Most seasons on Netflix')

        col = 'season_count'

        vc1 = netflix_tv[col].value_counts().reset_index()
        vc1 = vc1.rename(columns = {col : "count", "index" : col})
        vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
        vc1[col] = vc1[col].astype(int)
        vc1 = vc1.sort_values(col)

        trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
        data = [trace1]
        layout = go.Layout(legend=dict(x=0.1, y=1.1, orientation="h"))   

        fig = go.Figure(data, layout=layout)
        fig = fig.update_xaxes(title_text="Season")
        fig = fig.update_yaxes(title_text="Number of TV shows") 

        st.plotly_chart(fig)

        st.subheader('TV series with most seasons on Netflix')

        t = ['title','season_count']
        top = netflix_tv[t]
        top['season_count'] = top['season_count'].astype(int)
        top = top.sort_values(by='season_count', ascending=False)
        top20 = top[0:20]
        
        fig = top20.iplot(asFigure=True, kind='bar', fill=True, theme='pearl', orientation='h', xTitle='Number of seasons', yTitle='Title', x='title', y='season_count')
        st.plotly_chart(fig)


    # Content added over years by rating ----------------------------------------------------------------------
    if st.checkbox('Content by rating'):

        st.subheader('Content added by rating')

        col = "rating"

        vc1 = netflix_tv[col].value_counts().reset_index()
        vc1 = vc1.rename(columns = {col : "count", "index" : col})
        vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
        vc1 = vc1.sort_values(col)

        vc2 = netflix_movies[col].value_counts().reset_index()
        vc2 = vc2.rename(columns = {col : "count", "index" : col})
        vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
        vc2 = vc2.sort_values(col)

        trace1 = go.Bar(x=vc1[col], y=vc1["count"], name="TV Shows", marker=dict(color="#a678de"))
        trace2 = go.Bar(x=vc2[col], y=vc2["count"], name="Movies", marker=dict(color="#6ad49b"))
        data = [trace1, trace2]

        layout = go.Layout(legend=dict(x=0.1, y=1.1, orientation="h"))

        fig = go.Figure(data, layout=layout)
        fig.update_xaxes(title_text='Rating')
        fig.update_yaxes(title_text='Number of content added')

        st.plotly_chart(fig)

    # Content added over years by genre ---------------------------------------------------------------------
    if st.checkbox('Content by genre'):

        st.subheader('Content added by genre')

        col = "listed_in"

        categories = ", ".join(netflix_movies['listed_in']).split(", ")
        counter_list = Counter(categories).most_common(50)
        labels = [_[0] for _ in counter_list][::-1]
        values = [_[1] for _ in counter_list][::-1]
        trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))
        data = [trace1]

        layout = go.Layout(legend=dict(x=0.1, y=1.1, orientation="h"))

        fig = go.Figure(data, layout=layout)
        fig.update_xaxes(title_text='Number of content added')
        fig.update_yaxes(title_text='Genre')

        st.plotly_chart(fig)

    # Most appeared actors ------------------------------------------------------------------------------------
    if st.checkbox('Most appeared actors'):
        st.subheader('Most appeared actors in movies for the top 6 productive countries')
        def country_trace(country, flag = "movie"):
            netflix_data["from_us"] = netflix_data['country'].fillna("").apply(lambda x : 1 if country.lower() in x.lower() else 0)
            small = netflix_data[netflix_data["from_us"] == 1]
            if flag == "movie":
                small = small[small["duration"] != ""]
            else:
                small = small[small["season_count"] != ""]
            cast = ", ".join(small['cast'].fillna("")).split(", ")
            tags = Counter(cast).most_common(25)
            tags = [_ for _ in tags if "" != _[0]]

            labels, values = [_[0]+"  " for _ in tags], [_[1] for _ in tags]
            trace = go.Bar(y=labels[::-1], x=values[::-1], orientation="h", name="", marker=dict(color="#a678de"))
            return trace

        traces = []
        titles = ["United States", "","India","", "United Kingdom", "Canada","", "France","", "Japan"]
        for title in titles:
            if title != "":
                traces.append(country_trace(title))

        fig = make_subplots(rows=2, cols=5, subplot_titles=titles)
        fig.add_trace(traces[0], 1,1)
        fig.add_trace(traces[1], 1,3)
        fig.add_trace(traces[2], 1,5)
        fig.add_trace(traces[3], 2,1)
        fig.add_trace(traces[4], 2,3)
        fig.add_trace(traces[5], 2,5)

        fig.update_layout(height=1200, showlegend=False)
        st.plotly_chart(fig)

        st.subheader('Most appeared actors in TV series for the top 6 productive countries')
        traces = []
        titles = ["United States", "","India","", "United Kingdom", "Canada","", "France","", "Japan"]
        for title in titles:
            if title != "":
                traces.append(country_trace(title, flag="tv_shows"))

        fig = make_subplots(rows=2, cols=5, subplot_titles=titles)
        fig.add_trace(traces[0], 1,1)
        fig.add_trace(traces[1], 1,3)
        fig.add_trace(traces[2], 1,5)
        fig.add_trace(traces[3], 2,1)
        fig.add_trace(traces[4], 2,3)
        fig.add_trace(traces[5], 2,5)

        fig.update_layout(height=1200, showlegend=False)
        st.plotly_chart(fig)

    if st.checkbox('Directors with most content'):
        # Most directors -------------------------------------------------------------------------------------
        st.subheader('Directors with most content from the 3 most productive countries (USA, India, United Kingdom)')

        small = netflix_data[netflix_data["type"] == "Movie"]
        small = small[small["country"] == "United States"]

        col = "director"
        categories = ", ".join(small[col].fillna("")).split(", ")
        counter_list = Counter(categories).most_common(12)
        counter_list = [_ for _ in counter_list if _[0] != ""]
        labels = [_[0] for _ in counter_list][::-1]
        values = [_[1] for _ in counter_list][::-1]
        trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="orange"))

        data = [trace1]
        layout = go.Layout(title="USA", legend=dict(x=0.1, y=1.1, orientation="h"))
        fig = go.Figure(data, layout=layout)
        st.plotly_chart(fig)

        small = netflix_data[netflix_data["type"] == "Movie"]
        small = small[small["country"] == "India"]

        col = "director"
        categories = ", ".join(small[col].fillna("")).split(", ")
        counter_list = Counter(categories).most_common(12)
        counter_list = [_ for _ in counter_list if _[0] != ""]
        labels = [_[0] for _ in counter_list][::-1]
        values = [_[1] for _ in counter_list][::-1]
        trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="orange"))

        data = [trace1]
        layout = go.Layout(title='India', legend=dict(x=0.1, y=1.1, orientation="h"))
        fig = go.Figure(data, layout=layout)
        st.plotly_chart(fig)

        small = netflix_data[netflix_data["type"] == "Movie"]
        small = small[small["country"] == "United Kingdom"]

        col = "director"
        categories = ", ".join(small[col].fillna("")).split(", ")
        counter_list = Counter(categories).most_common(12)
        counter_list = [_ for _ in counter_list if _[0] != ""]
        labels = [_[0] for _ in counter_list][::-1]
        values = [_[1] for _ in counter_list][::-1]
        trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="orange"))

        data = [trace1]
        layout = go.Layout(title="United Kingdom", legend=dict(x=0.1, y=1.1, orientation="h"))
        fig = go.Figure(data, layout=layout)
        st.plotly_chart(fig)

if selected == 'Recommendations':

    st.title('Recommendations')
    expander = st.beta_expander("HOW IT WORKS")
    expander.write('*Content based* recommendations on the following factors:')
    expander.write('* Title')
    expander.write('* Cast')
    expander.write('* Director')
    expander.write('* Genre')
    expander.write('* Plot')

    # Contented Based Recommendation System---------------------------------------------------------
  
    #removing stopwords
    tfidf = TfidfVectorizer(stop_words='english')

    #Construct the required TF-IDF matrix by fitting and transforming the data
    features=['title','director','cast','listed_in','description']
    tfidf_matrix = tfidf.fit_transform(netflix_data[features])
    
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    filledna=netflix_data.fillna('')
    
    def clean_data(x):
        return str.lower(x.replace(" ", ""))

    filledna=filledna[features]

    for feature in features:
        filledna[feature] = filledna[feature].apply(clean_data)

    def create_soup(x):
        return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

    
    filledna['soup'] = filledna.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(filledna['soup'])

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
    filledna=filledna.reset_index()
    indices = pd.Series(filledna.index, index=filledna['title']).drop_duplicates()

    def get_recommendation(title, cosine_sim=cosine_sim):
        title=title.replace(' ','').lower()
        idx = indices[title]

        # Get the pairwsie similarity scores of all movies with that movie
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 10 most similar movies
        sim_scores = sim_scores[1:11]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores] 

        # Return the top 10 most similar movies
        return netflix_data['title'].iloc[movie_indices]

    answer = st.selectbox('Please select a movie you like', netflix_data.title)
    if st.button("Make recommendations"):
        st.write(get_recommendation(answer, cosine_sim2))

if selected == 'Informations':
    st.title('Find informations about movies you like')
    movie = st.selectbox('Informations for:', netflix_data.title)
    if st.button("Show informations"):
        description = netflix_data[netflix_data.title == movie].description
        try:
            st.subheader('Description')
            st.write(description.values[0])
        except:
            st.subheader('Description')
            st.write(f'Unfortunately there is no description for {movie}')
        
        director = netflix_data[netflix_data.title == movie].director
        if director.values[0] is not None:
            st.subheader('Director')
            st.write(director.values[0])
        else:
            st.subheader('Director')
            st.write(f'Unfortunately we do not know the director of {movie}')

        cast = netflix_data[netflix_data.title == movie].cast
        if cast.values[0] is not None:
            st.subheader('Cast')
            st.write(cast.values[0])
        else:
            st.subheader('Cast')
            st.write(f'Unfortunately we do not know the cast of {movie}')

        country = netflix_data[netflix_data.title == movie].country
        if country.values[0] is not None:
            st.subheader('Country of movie')
            st.write(country.values[0])
        else:
            st.subheader('Country')
            st.write(f'Unfortunately we do not know the country of {movie}')

        date_added = netflix_data[netflix_data.title == movie].date_added
        if not(np.isnat(date_added.values[0])):
            st.subheader('Date added on Netflix')
            st.write(date_added.values[0].astype('str')[:10])
        else:
            st.subheader('Date added on Netflix')
            st.write(f'Unfortunately we do not know the date that {movie} was added on Netflix')

        release_year = netflix_data[netflix_data.title == movie].release_year
        if not(np.isnan(release_year.values[0])):
            st.subheader('Year of release')
            st.write(release_year.values[0].astype('str')[:10])
        else:
            st.subheader('Year of release')
            st.write(f'Unfortunately we do not know the year that {movie} was released')

        rating = netflix_data[netflix_data.title == movie].rating
        if rating.values[0] is not None:
            st.subheader('Rating')
            st.write(rating.values[0])
        else:
            st.subheader('Rating')
            st.write(f'Unfortunately we do not know the rating of {movie}')

        genre = netflix_data[netflix_data.title == movie].listed_in
        try:
            st.subheader('Genre')
            st.write(genre.values[0])
        except:
            st.subheader('Genre')
            st.write(f'Unfortunately we do not know the genre of {movie}')
