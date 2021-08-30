from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, MeanShift
import shutil


class GeographicalClustering:

    def __init__(self):
        self.photo_df = pd.read_table(filepath_or_buffer='../data/Photo.csv',
                                      names=['photo_id', 'user_id', 'latitude', 'longitude', 'timestamp'],
                                      usecols=[0, 1, 2, 3, 4],
                                      sep=',').sample(100000, random_state=2)

        self.tag_df = pd.read_table('../data/Tag.csv', names=['photo_id', 'tag_id', 'tag_name'], sep=',')


    def show_infos(self):  # check data is non-null
        print(self.photo_df.info())
        print(self.photo_df.describe())
        print(self.tag_df.info())
        print(self.tag_df.describe())

    def join(self):
        print(self.photo_df.head())
        print(self.tag_df.head())
        return pd.merge(left=self.photo_df, right=self.tag_df, how='left', left_on='photo_id',
                               right_on='photo_id')

    def origin_img(self):
        fig = plt.figure()
        ax = fig.add_axes([.1, .1, 1, 1])
        joined_df = self.join()
        joined_df['tag_id'] = joined_df['tag_id'].apply(lambda x: 'black' if x == np.NAN else x)
        ax.scatter(joined_df['longitude'],
                   joined_df['latitude'],
                   c=joined_df['tag_id'].to_list(),
                   edgecolors='black', s=50)
        ax.set_xlabel('Longitude', family='Arial', fontsize=10)
        ax.set_ylabel('Latitude', family='Arial', fontsize=10)

        plt.title('Clustered GPS signals by origin', family='Arial', fontsize=14)
        plt.grid(which='major', color='#cccccc', alpha=0.45)
        plt.savefig('../img/origin.png', bbox_inches='tight')

    def plot_geo(self):
        _ = plt.plot(self.photo_df['longitude'], self.photo_df['latitude'], marker='.', linewidth=0, color='#128128')
        _ = plt.grid(which='major', color='#CCCCCC', alpha=0.45)
        _ = plt.title('Geographical distribution', family='Arial', fontsize=12)
        _ = plt.xlabel('longitude')
        _ = plt.ylabel('latitude')
        _ = plt.show()

    def no_job_plot(self):
        merged_left = pd.merge(left=self.photo_df, right=self.tag_df, how='left', left_on='photo_id',
                               right_on='photo_id')
        print(merged_left.head())

    def dbscan(self):

        # prepare data (longitude, latitude)
        data = self.photo_df[['longitude', 'latitude']]
        data = data.values.astype('float32', copy=False)

        # construct model
        # eps=1/6371. -> 1km
        model = DBSCAN(eps=0.5/6371., min_samples=20, algorithm='ball_tree', metric='haversine').fit(np.radians(data))

        # visualize
        outliers_df = self.photo_df[model.labels_ == -1]
        clusters_df = self.photo_df[model.labels_ != -1]

        colors = model.labels_
        colors_clusters = colors[colors != -1]
        colors_outliers = 'red'

        clusters = Counter(model.labels_)
        print(clusters)
        print(self.photo_df[model.labels_ == -1].head())
        print(self.photo_df[model.labels_ == 0].head())

        print('Number of clusters = {}'.format(len(clusters)-1))

        fig = plt.figure()

        ax = fig.add_axes([.2, .2, 3, 3])
        ax.scatter(clusters_df['longitude'], clusters_df['latitude'], c=colors_clusters, edgecolors='black', s=120)
        ax.scatter(outliers_df['longitude'], outliers_df['latitude'], c=colors_outliers, edgecolors='black', s=100)
        ax.set_xlabel('Longitude', family='Arial', fontsize=20)
        ax.set_ylabel('Latitude', family='Arial', fontsize=20)

        plt.title('Clustered GPS signals by DBSCAN', family='Arial', fontsize=25)
        plt.grid(which='major', color='#cccccc', alpha=0.45)
        plt.savefig('../img/dbscan.png', bbox_inches='tight')

    def mean_shift(self):
        # prepare data (longitude, latitude)
        data = self.photo_df[['longitude', 'latitude']]
        data = data.values.astype('float32', copy=False)

        model = MeanShift(bandwidth=0.07, bin_seeding=True, max_iter=100).fit(data)
        clusters_df = self.photo_df

        colors_clusters = model.labels_

        clusters = Counter(model.labels_)
        print(clusters)
        print(self.photo_df[model.labels_ == -1].head())
        print(self.photo_df[model.labels_ == 0].head())

        print('Number of clusters = {}'.format(len(clusters) - 1))

        fig = plt.figure()

        ax = fig.add_axes([.2, .2, 3, 3])
        ax.scatter(clusters_df['longitude'], clusters_df['latitude'], c=colors_clusters, edgecolors='black', s=120)
        ax.set_xlabel('Longitude', family='Arial', fontsize=20)
        ax.set_ylabel('Latitude', family='Arial', fontsize=20)

        plt.title('Clustered GPS signals by MeanShift', family='Arial', fontsize=25)
        plt.grid(which='major', color='#cccccc', alpha=0.45)
        plt.savefig('../img/mean_shift_100000.png', bbox_inches='tight')
        return model.labels_

    def divide_cluster(self, labels: list):
        for i in set(labels):
            for j in self.photo_df[labels == i]['photo_id']:
                try:
                    shutil.copyfile(f'../photos/{j}.jpg', f'../cluster/{i}/{j}.jpg')
                except FileNotFoundError:
                    pass

    def process(self):
        # self.show_infos()
        # self.plot_geo()
        # self.origin_img()
        # self.dbscan()
        labels = self.mean_shift()
        self.divide_cluster(labels)
        # self.join()
        # print(self.tag_df['tag_name'].unique())
