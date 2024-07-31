from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import silhouette_score

def for_visulize(df):
    kmodel = KMeans(n_clusters=5).fit(df[['Annual_Income','Spending_Score']])
    df['Cluster'] = kmodel.labels_
    df['Cluster'].value_counts()
    return df
    
def Elbow_method (df):
    k = range(3,9)
    K = []
    WCSS = []
    for i in k:
        kmodel = KMeans(n_clusters=i).fit(df[['Annual_Income','Spending_Score']])
        wcss_score = kmodel.inertia_
        WCSS.append(wcss_score)
        K.append(i)

    wss = pd.DataFrame({'cluster': K, 'WSS_Score':WCSS})
    return wss

def silhouette_method(df):
    k = range(3,9) # to loop from 3 to 8
    K = []         # to store the values of k
    ss = []        # to store respective silhouetter scores
    for i in k:
        kmodel = KMeans(n_clusters=i,).fit(df[['Annual_Income','Spending_Score']], )
        ypred = kmodel.labels_
        sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], ypred)
        K.append(i)
        ss.append(sil_score)
        
    wsss = pd.DataFrame({'cluster': K, 'Silhouette_Score':ss})
    return wsss