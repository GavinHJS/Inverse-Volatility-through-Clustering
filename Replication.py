# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:46:58 2024

@author: Gavin
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def wrapper_data():
    dir_path = './Data'
    
    all_files = os.listdir(dir_path)
    
    
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    
    all_data = pd.DataFrame()
    
    
    for csv_file in csv_files:
        file_path = os.path.join(dir_path, csv_file)
        current_data = pd.read_csv(file_path)
        all_data = pd.concat([all_data, current_data], ignore_index=True)
    all_data['Analysis Date'] = pd.to_datetime(all_data['Analysis Date'], format='%d/%m/%Y')
    all_data = all_data[["Analysis Date" , "Asset Name" , "Price" , "Return (%)" ]]
    all_data["Price"] = all_data["Price"].astype(float)
    all_data["Return (%)"] = all_data["Return (%)"].apply(lambda x: x[:-1]).astype(float)/100
    return all_data

def get_correlation(data,rolling_window = 20):
    data = data.drop_duplicates(subset=['Analysis Date', 'Asset Name'])
    df_pivoted = data.pivot(index="Analysis Date", columns="Asset Name", values="Return (%)")
    df_pivoted = df_pivoted.fillna(0)

    correlation_matrix = df_pivoted.rolling(window=rolling_window).corr()
    correlation_matrix = correlation_matrix.to_numpy()

    return  correlation_matrix
    

def get_distancematrix(correlation_matrix):
    distance_matrix = squareform(pdist(correlation_matrix, 'euclidean'))
    return (distance_matrix)

def find_optimal_clusters(distance_mat, max_clusters=10):
    silhouette_scores = {}
    for n_clusters in range(2, max_clusters + 1):
        for linkage in ['ward', 'complete', 'single']:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = clusterer.fit_predict(distance_mat)
            score = silhouette_score(distance_mat, labels, metric='precomputed')
            silhouette_scores[(linkage, n_clusters)] = score
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(distance_mat)
        score = silhouette_score(distance_mat, labels, metric='precomputed')
        silhouette_scores[('kmeans', n_clusters)] = score
    return silhouette_scores

def compute_inverse_volatility_weights(returns):
    volatilities = returns.std()
    inverse_volatility = 1 / volatilities
    normalized_weights = inverse_volatility / inverse_volatility.sum()
    return normalized_weights

def cluster_assets(data, best_setup, distance_matrix):
    method, n_clusters = best_setup
    if method in ['ward', 'complete', 'single']:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    elif method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    labels = clustering.fit_predict(distance_matrix)
    return labels

def construct_portfolio(data, labels):
    data['Cluster'] = labels
    cluster_portfolios = {}
    for cluster in set(labels):
        cluster_data = data[data['Cluster'] == cluster]
        cluster_returns = cluster_data.pivot(index="Analysis Date", columns="Asset Name", values="Return (%)").fillna(0)
        weights = compute_inverse_volatility_weights(cluster_returns)
        cluster_portfolios[cluster] = weights
    return cluster_portfolios

if __name__ =="__main__":
    
    data = wrapper_data()
    correl = get_correlation(data)
    distance_matrix = get_distancematrix(correl)
    silhouette_scores = find_optimal_clusters(distance_matrix)
    best_setup = max(silhouette_scores, key=silhouette_scores.get)
    labels = cluster_assets(data, best_setup, distance_matrix)
    cluster_portfolios = construct_portfolio(data, labels)
    print(cluster_portfolios)