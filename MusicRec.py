#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import nltk
import pyspark
import findspark
import pandas as pd
from IPython.display import HTML
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, functions as F 

from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import ClusteringEvaluator

# a_dict = {}

# for name in list(df["Processed Playlist name"]):
#     a_dict[name] = a_dict.get(name,0)+1
    
# a_df = pd.DataFrame(a_dict.items(),columns=["Word","Count"])
# a_df = a_df.sort_values(["Count"], ascending=False)

# def reduce_df(number):
#     play = a_df.iloc[number-1,0]
#     indx = 0
#     for i in range(df.shape[0]):
#         if df.iloc[i,4]==play:
#             indx=i
        
#     df = df.iloc[:indx]
#     return df

# df = reduce_df(250)
# df.to_csv("Spotify Music Data.csv", index=False)


# In[2]:


df = pd.read_csv("Spotify Music Data.csv")
df.head()


# In[3]:


stop_words = stopwords.words('english') + stopwords.words('spanish')
stop_words.remove("out")

for word in ["original", "motion", "picture", "soundtrack", "artist", "music", "song", "best", "part", "album", "mix", "edit"]:
    stop_words.append(word)


# In[4]:


play_ct = len(list(df["Playlist name"].unique()))
user_ct = len(list(df["User ID"].unique()))
track_ct = len(list(df["Track name"].unique()))
print(f"Data Size\t\t= {df.shape[0]}\nNumber of Tracks\t= {track_ct}\nNumber of Users\t\t= {user_ct}\nNumber of Playlists\t= {play_ct}")


# There are currently 4 million rows of data comprising of 11,000+ users from whom nearly 7,000 playlists with a little over one million tracks are present

# In[5]:


def pre_processing(attribute):
    
    processed_names = []
    
    for ply_name in list(df[attribute]):
        ply_name = re.sub("\d+", "", ply_name)
        ply_name = re.sub(r'[^\w\s]', ' ', ply_name)
        ply_name = [WNL.lemmatize(word) for word in ply_name.lower().split() if word not in stop_words]
        ply_name = " ".join(ply_name)
        
        if ply_name != "":
            processed_names.append(ply_name)
        else:
            processed_names.append("Number/Stopwords")
        
    return processed_names


# In[6]:


WNL = nltk.WordNetLemmatizer()
df["Processed Playlist name"] = pre_processing("Playlist name")
df.head()


# In[7]:


len(list(df["Processed Playlist name"].unique()))


# In[8]:


dummies = pd.get_dummies(df["Processed Playlist name"])
dummies.head()


# In[9]:


cols = []
for i in range(1,11):
    cols.append("Component "+str(i))
    
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(dummies)
PCA_df = pd.DataFrame(data = principalComponents, columns = cols)

# PCA_df.to_csv("PCA.csv", index=False)

PCA_df.head()


# In[10]:


# Initating a Spark session
findspark.init()
findspark.find()
conf = pyspark.SparkConf().setAppName('appName').setMaster('local')
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession(sc)


# In[11]:


#Converting to a spark DataFrame
# spark_df = spark.createDataFrame(PCA_df)

spark_df = spark.read.csv("PCA.csv",header=True, inferSchema=True)

spark_df.show()


# In[12]:


evaluator = ClusteringEvaluator()

# Gathering the input features under a single column named "features" using VectorAssembler
vecAssembler = VectorAssembler(inputCols=['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 
                                          'Component 6', 'Component 7', 'Component 8', 'Component 9', 'Component 10'],
                               outputCol="features")

spark_df = vecAssembler.transform(spark_df)
spark_df.show()


# In[13]:


# Creating a new dataframe to store Silhouette measurements for different K values
k_df = pd.DataFrame(columns = ['K', 'Silhouette'])

# Running 10 interations for
for i,k_val in enumerate(range(2,26)):
    
    # K-Means object
    kmeans = KMeans(k=k_val, seed=1)
    
    # Fitting the model
    model = kmeans.fit(spark_df.select('features'))
    
    # Transforming the dataframe to include the cluster prediction column
    transformed = model.transform(spark_df)
    
    # Silhouette measurement
    silhouette = evaluator.evaluate(transformed)
    
    # Appending value to the dataframe
    k_df = k_df.append({'K' : k_val, 'Silhouette' : round(silhouette,4)}, ignore_index = True)
    
    print(f'Clustering for K={k_val} completed')
    
k_df.to_csv("Silhouette Measures.csv",index=False)


# In[14]:


k_df = k_df.sort_values(["Silhouette"], ascending=False)
print(f"Best K = {k_df.iloc[0,0]} with Silhouette = {k_df.iloc[0,1]}")


# In[15]:


# kmeans = KMeans(k=14, seed=1)
# model = kmeans.fit(spark_df.select('features'))
# transformed_final = model.transform(spark_df)

# final_df = transformed_final.toPandas()
# final_df = final_df.rename(columns={"prediction":"Cluster"})
# final_df = df[['User ID', 'Artist Name', 'Track name', 'Playlist name']].join(final_df["Cluster"])
###############################################################################################
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=14).fit(PCA_df)
final_df = df[['User ID', 'Artist Name', 'Track name', 'Playlist name']]
final_df["Cluster"] = list(kmeans.labels_)
###############################################################################################
final_df.head()


# In[16]:


def song_suggestion(prim_user):
    cluster = list(final_df.loc[final_df["User ID"] == prim_user,"Cluster"])[0]
    
    users_in_cluster = list(set(final_df.loc[final_df["Cluster"] == cluster,"User ID"]))
    users_in_cluster.remove(prim_user)
    
    sim_df = pd.DataFrame(columns=["User ID", "Similarity"])
    
    set_prim_user = set(final_df.loc[final_df["User ID"] == prim_user,"Track name"])
    
    for sec_user in users_in_cluster:
        set_sec_user = set(final_df.loc[final_df["User ID"] == sec_user,"Track name"])
        
        similarity = len(set_prim_user & set_sec_user) / len(set_prim_user | set_sec_user)
        
        sim_df = sim_df.append({"User ID":sec_user, "Similarity":round(similarity,4)}, ignore_index = True)
        
    sim_df.sort_values(["Similarity"], ascending=False)
    
    songs_suggestions = {}
    
    for i in range(sim_df.shape[0]):
        suggestion_df = final_df.loc[final_df["User ID"] == sim_df.iloc[i,0],["Artist Name", "Track name"]]
        
        for artist, track in zip(list(suggestion_df["Artist Name"]) , list(suggestion_df["Track name"])):
            
            if len(songs_suggestions)==3:
                songs_suggestions_df = HTML(pd.DataFrame(songs_suggestions.items(), columns=["Track","Artist"]).to_html(index=False))
                return songs_suggestions_df
            
            elif track not in set_prim_user:
                songs_suggestions[track]=artist


# In[17]:


prim_user = "673db58dbfdec08eb992a3a072dbf24b"
song_suggestion(prim_user)


# In[ ]:




