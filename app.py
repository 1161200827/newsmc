# Core Pkgs
import streamlit as st 

# EDA Pkgs
import pandas as pd 
import numpy as np 

# Data Viz Pkg
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 
import pickle
import dill
from PIL import Image
dill.dumps('foo')
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from wordcloud import WordCloud
from PIL import Image
from SPARQLWrapper import SPARQLWrapper
from streamlit_agraph import agraph, TripleStore, Config
st.set_option('deprecation.showPyplotGlobalUse', False)
import networkx as nx
from pyvis.network import Network
import community
from community import community_louvain
import streamlit.components.v1 as components

def main():
    """Semi Automated ML App with Streamlit """
    activities = ["Matrics Monitored","Data Visualization", "About Us"]  

    choice = st.sidebar.selectbox("Select Activities", activities)
    companyCol = ["blue","orange","green","red"]
    
    if choice == 'About Us':
       st.title('Social Media Computing Project Members')
       about = pd.read_csv('aboutus.csv')
       st.table(about)
        
    if choice == 'Matrics Monitored':
        
        st.title("Social Media Computing Assignment")
        st.write("In the project, we have decided to run an analysis on four different competing companies which are OMEGA (@omegawatches), ROLEX (@ROLEX), Daniel Wellington (@itisDW), and Swatch US (@SwatchUS) through Twitter API.")
        st.write("Based on the data crawled through the Twitter API in February 2021, several metrics can be used to monitor the best campaigns for all brands. As a result, Daniel wellington success run a campaign #danielwellington which the engagement rate is higher among the competitor.")
        
        image = Image.open('logo.jpeg')
        new_image = image.resize((600, 150))
        st.image(new_image)
        

        
        social = pd.read_csv('social_media.csv')
        fig = px.bar(social, x="domain", y="value", color="type", title="Social Media Platform Analysis")
        st.plotly_chart(fig)
        
        df = pd.read_csv('audience_size.csv')
        #df
        fig = px.bar(df, x="company", y="count", color="type", title="Audience Size")
        st.plotly_chart(fig)
        
        animals=['OMEGAOfficialTimekeeper@OMEGA', 'perpetual@ROLEX',
            'danielwellington@ITISDW','swatchmytime@SWATCH']
            
        fig = go.Figure([go.Bar(x = animals, y = [0.2139, 0.0914, 7.7447, 0.0377])])
        st.subheader("Best campaign across all the brands")
        st.plotly_chart(fig)
        
        social = pd.read_csv('camp_categories.csv')
        fig = px.bar(social, x="domain", y="engagement_rate", color="campaign", title="Campaign")
        st.plotly_chart(fig)
        
    if choice == 'Data Visualization':   
        
        activities = ["Audience Country Spread","Engagement Rate","Campaign Categories","Favourite And Retweet Count","Word Cloud","Network Graph"]  
        choice = st.sidebar.selectbox("Select Activities",activities)
    
        if choice == 'Audience Country Spread':
            st.title("Audience Country Spread") 
            
            loc_data = ["omega_loc", "rolex_loc", "itisdw_loc", "swatch_loc"]
            comp_name = ["OMEGA", "ROLEX", "ITISDW", "SWATCH"]
          
            for k in range(4):
                loc_set = pd.read_csv("%s.csv" % loc_data[k])
                loc_chart = px.bar(loc_set,x='Followers(%)',y='Country',
                title="%s Favourite Audience country spread" % comp_name[k],
                text='Followers(%)',
                color_continuous_scale=px.colors.sequential.Sunset,
                color='Country',
                orientation='h'
                )
              
                st.plotly_chart(loc_chart)
                    
        if choice == 'Campaign Categories':
            
            st.title("Campaign Categories") 
            
            
            camp_cate_opt = ["Total Campaign Engagement Rate","Total Tweeted Campaign","Average Campaign Engagement"]
            camp_cate = st.selectbox("Select Campaign", camp_cate_opt) 
            companyName = ["omega","rolex","itisdw","swatch"]

            if camp_cate == 'Total Campaign Engagement Rate':
                for k in range(4):
                    companyChoose = companyName[k]
                    camp_eng = pd.read_csv('campaign_engagement.csv').query('company == @companyChoose')
                    fig = px.bar(camp_eng, x="campaign", y="rate", title=companyChoose)
                    fig.update_traces(marker_color=companyCol[k])
                    st.plotly_chart(fig)
                
                
            if camp_cate == 'Total Tweeted Campaign':
                for k in range(4):
                    companyChoose = companyName[k]
                    camp_eng = pd.read_csv('campaign_engagement.csv').query('company == @companyChoose')
                    fig = px.bar(camp_eng, x="campaign", y="count", title=companyChoose)
                    fig.update_traces(marker_color=companyCol[k])
                    st.plotly_chart(fig)
               
            if camp_cate == 'Average Campaign Engagement':
                for k in range(4):
                    companyChoose = companyName[k]
                    camp_eng = pd.read_csv('campaign_engagement.csv').query('company == @companyChoose')
                    fig = px.bar(camp_eng, x="campaign", y="rate", title=companyChoose)
                    fig.update_traces(marker_color=companyCol[k])
                    st.plotly_chart(fig)

        if choice == 'Engagement Rate':

            st.title("Engagement Rate")
            eng_rate_opt = ["ALL","omega","rolex","itisdw", "swatch"]
            eng_rate_val = st.selectbox("Select Brand", eng_rate_opt)
            

            if eng_rate_val == 'ALL':
                eng_freq = pd.read_csv('engagement_rate.csv')
                fig = px.line(eng_freq, x="week", y="count", color='company', title="Engagement rate")
                st.plotly_chart(fig)
                
                tweet_freq = pd.read_csv('tweet_freq.csv')
                fig = px.line(tweet_freq, x="week", y="count", color='company', title="Tweet Frequency")
                st.plotly_chart(fig)
                
            else:
                eng_freq = pd.read_csv('engagement_rate.csv')
                eng_freq = eng_freq.query('company == @eng_rate_val')
                fig = px.line(eng_freq, x="week", y="count", 
                   title="%s Engagement Rate" % eng_rate_val)
                st.plotly_chart(fig)

                tweet_freq = pd.read_csv('tweet_freq.csv')                
                tweet_freq = tweet_freq.query('company == @eng_rate_val')
                fig = px.line(tweet_freq, x="week", y="count", 
                   title="%s Frequency Tweet" % eng_rate_val)
                st.plotly_chart(fig)
                
        if choice == 'Favourite And Retweet Count':

            st.title("Favourite And Retweet Count")   
      
            fav_opt = ["Favourite", "Retweet"]  
            fav_selc = st.selectbox("Select Activities", fav_opt)
            
            fav_data = ["omega_camp", "rolex_camp", "itisdw_camp", "swatch_camp"]
            comp_name = ["OMEGA", "ROLEX", "ITISDW", "SWATCH"]
             
            if fav_selc == 'Favourite':
                for i in range(4):
                    data = px.bar(pd.read_csv("%s.csv" % fav_data[i]),x='campaign',y='likes_sum',
                        title="%s Favourite Count" % comp_name[i])
                    st.plotly_chart(data)
       
            if fav_selc == 'Retweet':
                for i in range(4):
                    data = px.bar(pd.read_csv("%s.csv" % fav_data[i]),x='campaign',y='retweet_sum',
                        title="%s Retweet Count" % comp_name[i])
                    st.plotly_chart(data)

        if choice == 'Word Cloud':  
        
            image_mask = np.array(Image.open("watch.jpg"))
            word_cloud = pd.read_csv('word_cloud.csv')
        
            st.title("Word Cloud")         
            brand = ["omega","rolex","itisdw", "swatch"]
            brand_choice = st.selectbox("Select Brand", brand)
            
            word = word_cloud.query('(company == @brand_choice)')
            camp = word.campaign.to_numpy()
            
            if brand_choice == 'omega':
                camp_opt = ['all','DeVille','ValentinesDay','OMEGAOfficialTimekeeper']
                
            if brand_choice == 'rolex':
                camp_opt = ['all','perpetual','reloxfamily']
                
            if brand_choice == 'itisdw':
                camp_opt = ['all','dwgiftsoflove','danielwellington','layzhang']
                
            if brand_choice == 'swatch':
                camp_opt = ['all','timeiswhatyoumakeofit','swatchwithlove','swatchmytime']
            
            camp_choice = st.selectbox("Select Campaign", camp_opt)
            word = word_cloud.query('(company == @brand) & (campaign == @camp_choice)')['word'].values[0]

            # generate word cloud
            wc = WordCloud(background_color="white", max_words=2000, mask=image_mask)
            wc.generate(word)

            # plot the word cloud
            plt.figure(figsize=(8,6), dpi=120)
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()
                 
        if choice == 'Network Graph':
        
            st.title("Network Graph")
            brand = ["omega","rolex","itisdw","swatch"]
            brand_choice = st.selectbox("Select Brand", brand)

            st.title("Measurement of Centrality")
            net_opt = ["degree","betweenness","eigenvector", "closeness"]
            net_selc = st.selectbox("Select Variable", net_opt)
            
            net_data = pd.read_csv('table_centrality.csv').query('domain == @brand_choice & variable == @net_selc')
            st.table(net_data)
            
            nodelist = pd.read_csv(brand_choice+'_nodelist.csv')
            edgelist = pd.read_csv(brand_choice+'_edgelist.csv')

            G = nx.Graph()
            node_names = list(nodelist['screen_name'])

            edges = []
            for idx, row in edgelist.iterrows():
                edges.append(tuple(row))

            G.add_nodes_from(node_names)
            G.add_edges_from(edges)

            subgraph = sorted(nx.connected_components(G), key=len, reverse=True)[0]
            subgraph = G.subgraph(subgraph)

            communities = community_louvain.best_partition(G)
            values = [communities.get(node) for node in G.nodes()]
            plt.figure(figsize=(20,20))
            nx.draw_spring(G, cmap=plt.get_cmap('Spectral'), node_color = values, node_size=50, with_labels=False)
            
     
            st.title("Visualization Network Graph")
            
            
            G.add_nodes_from(node_names)
            G.add_edges_from(edges)
            st.write("Number of nodes: %s" % G.number_of_nodes())
            st.write("Number of edges: %s" % G.number_of_edges())
            st.write("Network density: %s" % nx.density(G))

            st.pyplot()
            
            #nt = Network("800px", "1500px")
            #nt.from_nx(subgraph)
            st.title("Interactive Visualization Network Graph")
            st.write("Please drag and scroll the graph with the mouse to show more detailed information")
            net_graph = "%s_network_graph.html" % brand_choice
            #st.write(net_graph)
            #components.iframe(net_graph, width= 800, height=800)
            HtmlFile = open(net_graph, 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=1000, width=650)

           
                    
if __name__=='__main__':
    main()