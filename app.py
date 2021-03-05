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

from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, silhouette_samples, silhouette_score

def main():
    """Semi Automated ML App with Streamlit """
    activities = ["Matrics Monitored","Data Visualization"]  
    
    choice = st.sidebar.selectbox("Select Activities",activities)

    
    if choice == 'Matrics Monitored':
        
        st.title("Social Media Computing Assignment")
       
        df = pd.read_csv('audience_size.csv')
       # df
        fig = px.bar(df, x="company", y="count", color="type", title="Audience Size")
        st.plotly_chart(fig)
        
        
    if choice == 'Data Visualization':   
        
        activities = ["Audience Country Spread","Campaign Categories","Favourite And Retweet Count", "Frequency Tweets","Word Map","Network Graph"]  
        choice = st.sidebar.selectbox("Select Activities",activities)
    
        if choice == 'Audience Country Spread':
            st.title("Audience Country Spread") 
            brand = ["Omega","Rolex","Daniel Wellington", "SwatchUS"]
            brand_choice = st.sidebar.selectbox("Select Brand",brand)
            if brand_choice == 'Omega':
            
                omega_loc = pd.read_csv('omega_loc.csv')
                omega_country = px.bar(omega_loc,x='Followers(%)',y='Country',
                title='Omega - Audience country spread',
                text='Followers(%)',
                orientation='h')
                st.plotly_chart(omega_country)
         
            if brand_choice == 'Rolex':
            
                rolex_loc = pd.read_csv('rolex_loc.csv')
                rolex_country = px.bar(rolex_loc,x='Followers(%)',y='Country',
                title='Rolex - Audience country spread',
                text='Followers(%)',
                orientation='h')
                st.plotly_chart(rolex_country)
        
            if brand_choice == 'Daniel Wellington':
            
                DanielWellington_loc = pd.read_csv('itisdw_loc.csv')
                DanielWellington_country = px.bar(DanielWellington_loc,x='Followers(%)',y='Country',
                title='Daniel Wellington - Audience country spread',
                text='Followers(%)',
                orientation='h')
                st.plotly_chart(DanielWellington_country)
            
            if brand_choice == 'SwatchUS':
            
                swatch_loc = pd.read_csv('swatch_loc.csv')
                swatch_country = px.bar(swatch_loc,x='Followers(%)',y='Country',
                title='SwatchUS - Audience country spread',
                text='Followers(%)',
                orientation='h')
                st.plotly_chart(swatch_country)
            
            
  

        
        if choice == 'Campaign Categories':
            
            st.title("Campaign Categories")   
      
            brand = ["Omega","Rolex","Daniel Wellington", "SwatchUS"]
            brand_choice = st.sidebar.selectbox("Select Brand",brand)
            if brand_choice == 'Omega':
            
                company = "omega"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)
                
            if brand_choice == 'Rolex':
            
                company = "rolex"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)
        
            if brand_choice == 'Daniel Wellington':
            
                company = "itisdw"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)
            
            if brand_choice == 'SwatchUS':
            
                company = "swatch"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)

        if choice == 'Favourite And Retweet Count':
    

            st.title("Favourite Count")   
      
            omega_camp = pd.read_csv('omega_camp.csv')
            omega_camp = px.bar(omega_camp,x='campaign',y='likes_sum',
            title='Omega Favourite Count')
            st.plotly_chart(omega_camp)
       
            rolex_camp = pd.read_csv('rolex_camp.csv')
            rolex_camp = px.bar(rolex_camp,x='campaign',y='likes_sum',
            title='Rolex Favourite Count')
            st.plotly_chart(rolex_camp)
            
            itisdw_camp = pd.read_csv('itisdw_camp.csv')
            itisdw_camp = px.bar(itisdw_camp,x='campaign',y='likes_sum',
            title='Daniel Wellington Favourite Count')
            st.plotly_chart(itisdw_camp)
        
       
            swatch_camp = pd.read_csv('swatch_camp.csv')
            swatch_camp = px.bar(swatch_camp,x='campaign',y='likes_sum',
            title='SwatchUS Favourite Count')
            st.plotly_chart(swatch_camp)
 
            st.title("Retweet Count")
        
            omega_camp = pd.read_csv('omega_camp.csv')
            omega_camp = px.bar(omega_camp,x='campaign',y='retweet_sum',
            title='Omega Retweet Count')
            st.plotly_chart(omega_camp)
          
            rolex_camp = pd.read_csv('rolex_camp.csv')
            rolex_camp = px.bar(rolex_camp,x='campaign',y='retweet_sum',
            title='Rolex Retweet Count')
            st.plotly_chart(rolex_camp)
         
            itisdw_camp = pd.read_csv('itisdw_camp.csv')
            itisdw_camp = px.bar(itisdw_camp,x='campaign',y='retweet_sum',
            title='Daniel Wellington Retweet Count')
            st.plotly_chart(itisdw_camp)
        
            swatch_camp = pd.read_csv('swatch_camp.csv')
            swatch_camp = px.bar(swatch_camp,x='campaign',y='retweet_sum',
            title='SwatchUS Retweet Count')
            st.plotly_chart(swatch_camp)
    
        if choice == 'Frequency Tweets':  
        
            st.title("Frequency Tweet")   
      
            brand = ["All","Omega","Rolex","Daniel Wellington", "SwatchUS"]
            brand_choice = st.sidebar.selectbox("Select Brand",brand)
            
            if brand_choice == 'Omega':
            
                company = "omega"      
                tweet_freq = pd.read_csv('tweet_freq.csv')                
                tweet_freq = tweet_freq.query('company == @company')
                fig = px.line(tweet_freq, x="week", y="count", title="Tweet Frequency")
                st.plotly_chart(fig)

                company = "omega" 
                eng_freq = pd.read_csv('engagement_rate.csv')
                eng_freq = eng_freq.query('company == @company')
                fig = px.line(eng_freq, x="week", y="count", title="Engagement rate")
                st.plotly_chart(fig)
                
            if brand_choice == 'Rolex':
            
                company = "rolex"      
                tweet_freq = pd.read_csv('tweet_freq.csv')                
                tweet_freq = tweet_freq.query('company == @company')
                fig = px.line(tweet_freq, x="week", y="count", title="Tweet Frequency")
                st.plotly_chart(fig)

                company = "rolex" 
                eng_freq = pd.read_csv('engagement_rate.csv')
                eng_freq = eng_freq.query('company == @company')
                fig = px.line(eng_freq, x="week", y="count", title="Engagement rate")
                st.plotly_chart(fig)
        
            if brand_choice == 'Daniel Wellington':
            
                company = "itisdw"      
                tweet_freq = pd.read_csv('tweet_freq.csv')                
                tweet_freq = tweet_freq.query('company == @company')
                fig = px.line(tweet_freq, x="week", y="count", title="Tweet Frequency")
                st.plotly_chart(fig)

                company = "itisdw" 
                eng_freq = pd.read_csv('engagement_rate.csv')
                eng_freq = eng_freq.query('company == @company')
                fig = px.line(eng_freq, x="week", y="count", title="Engagement rate")
                st.plotly_chart(fig)
            
            if brand_choice == 'SwatchUS':
            
                company = "swatch"      
                tweet_freq = pd.read_csv('tweet_freq.csv')                
                tweet_freq = tweet_freq.query('company == @company')
                fig = px.line(tweet_freq, x="week", y="count", title="Tweet Frequency")
                st.plotly_chart(fig)

                company = "swatch" 
                eng_freq = pd.read_csv('engagement_rate.csv')
                eng_freq = eng_freq.query('company == @company')
                fig = px.line(eng_freq, x="week", y="count", title="Engagement rate")
                st.plotly_chart(fig)
        
            if brand_choice == 'All':
                tweet_freq = pd.read_csv('tweet_freq.csv')
                fig = px.line(tweet_freq, x="week", y="count", color='company', title="Tweet Frequency")
                st.plotly_chart(fig)

                eng_freq = pd.read_csv('engagement_rate.csv')
                fig = px.line(eng_freq, x="week", y="count", color='company', title="Engagement rate")
                st.plotly_chart(fig)
                
        if choice == 'Word Map':  
        
            image_mask = np.array(Image.open("watch.jpg"))
            word_cloud = pd.read_csv('word_cloud.csv')
        
            st.title("Word Map")         
            brand = ["Omega","Rolex","Daniel Wellington", "SwatchUS"]
            brand_choice = st.sidebar.selectbox("Select Brand",brand)
            
            word = word_cloud.query('(company == @brand_choice)')
            camp = word.campaign.to_numpy()
            
            


            if brand_choice == 'Omega':
            
                camp = ['all','DeVille','ValentinesDay','OMEGAOfficialTimekeeper']
                camp_choice = st.sidebar.selectbox("Select Brand",camp)
                
                company = 'omega'
                campaign = "all"
                word = word_cloud.query('(company == @company) & (campaign == @camp_choice)')['word'].values[0]

                # generate word cloud
                wc = WordCloud(background_color="white", max_words=2000, mask=image_mask)
                wc.generate(word)
             
                
                # plot the word cloud
                plt.figure(figsize=(8,6), dpi=120)
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot()
                

            if brand_choice == 'Rolex':
            
                camp = ['all','perpetual','reloxfamily']
                camp_choice = st.sidebar.selectbox("Select Brand",camp)
                
                company = 'rolex'
                campaign = "all"
                word = word_cloud.query('(company == @company) & (campaign == @camp_choice)')['word'].values[0]

                # generate word cloud
                wc = WordCloud(background_color="white", max_words=2000, mask=image_mask)
                wc.generate(word)

                # plot the word cloud
                plt.figure(figsize=(8,6), dpi=120)
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot()
                
            if brand_choice == 'Daniel Wellington':
            
                camp = ['all','dwgiftsoflove','danielwellington','layzhang']
                camp_choice = st.sidebar.selectbox("Select Brand",camp)
                
                company = 'itisdw'
                campaign = "all"
                word = word_cloud.query('(company == @company) & (campaign == @camp_choice)')['word'].values[0]

                # generate word cloud
                wc = WordCloud(background_color="white", max_words=2000, mask=image_mask)
                wc.generate(word)

                # plot the word cloud
                plt.figure(figsize=(8,6), dpi=120)
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot()  

            if brand_choice == 'SwatchUS':
            
                camp = ['all','timeiswhatyoumakeofit','swatchwithlove','swatchmytime']
                camp_choice = st.sidebar.selectbox("Select Brand",camp)
                 
                company = 'swatch'
                campaign = "all"
                word = word_cloud.query('(company == @company) & (campaign == @camp_choice)')['word'].values[0]

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
      
            brand = ["Omega","Rolex","Daniel Wellington", "SwatchUS"]
            brand_choice = st.sidebar.selectbox("Select Brand",brand)
            if brand_choice == 'Omega':
            
                st.title("Graph Example")
                config = Config(height=500, width=700, nodeHighlightBehavior=True, highlightColor="#F7A7A6", directed=True,
                  collapsible=True)
                if query_type=="Inspirationals":
                    st.subheader("Inspirationals")
                    with st.spinner("Loading data"):
                        store = get_inspired()
                        st.write(len(store.getNodes()))
                    st.success("Done")
                    agraph(list(store.getNodes()), (store.getEdges() ), config)
                
            if brand_choice == 'Rolex':
            
                company = "rolex"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)
        
            if brand_choice == 'Daniel Wellington':
            
                company = "itisdw"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)
            
            if brand_choice == 'SwatchUS':
            
                company = "swatch"
                camp_eng = pd.read_csv('campaign_engagement.csv')
                camp_eng = camp_eng.query('company == @company')
                fig = px.bar(camp_eng, x="campaign", y="avg", title="Average Campaign Engagement")
                st.plotly_chart(fig)
                
                fig = px.bar(camp_eng, x="campaign", y="count", title="Total Tweeted Campaign")
                st.plotly_chart(fig)
                       
                fig = px.bar(camp_eng, x="campaign", y="rate", title="Total Campaign Engagement Rate")
                st.plotly_chart(fig)
    
if __name__=='__main__':
    main()