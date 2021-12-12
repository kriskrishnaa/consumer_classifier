from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Technical test sample data.ods", engine="odf")

selected_columns = ['has_gender', 'has_first_name',
       'has_last_name', 'has_email', 'has_dob', 'account_age',
       'account_last_updated', 'app_downloads',
       'unique_offer_clicked', 'total_offer_clicks', 'unique_offer_rides',
       'total_offer_rides', 'avg_claims', 'min_claims', 'max_claims',
       'total_offers_claimed']

train_df = df[selected_columns]

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(train_df)
    train_df["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.savefig("SSE.png", dpi=150)
plt.show()


## Silhouette Coefficient Method
for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(train_df)
    label = kmeans.labels_
    sil_coeff = silhouette_score(train_df, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
