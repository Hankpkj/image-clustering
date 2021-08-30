import os
import shutil

from keras.preprocessing.image import load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
import numpy as np
import pickle

from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA


class VisualClustering:

    def __init__(self, gc):
        self.geo_cluster = gc
        self.path = f'../cluster-{gc}'
        os.chdir(self.path)

        self.photos = []

        with os.scandir(self.path) as files:
            for file in files:
                if file.name.endswith('.jpg'):
                    self.photos.append(file.name)

        self.img = load_img(self.photos[0], target_size=(224, 224))
        self.img = np.array(self.img)

        # load model ( pre-trained )
        self.model = VGG16()
        self.model = Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

    @staticmethod
    def extract_features(file, model):
        img = load_img(file, target_size=(224, 224))

        img = np.array(img).reshape(1, 224, 224, 3)

        preprocessed = preprocess_input(img)

        features = model.predict(preprocessed, use_multiprocessing=True)

        return features

    def process(self):
        data = {}
        p = f"../model/gc-{self.geo_cluster}-to-vc-features.pkl"
        for pt in self.photos:
            try:
                feat = VisualClustering.extract_features(pt, self.model)
                data[pt] = feat
            except:
                with open(p, 'wb') as file:
                    pickle.dump(data, file)

        filenames = np.array(list(data.keys()))

        feat = np.array(list(data.values())).reshape(-1, 4096)

        pca = PCA(n_components=20, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)

        # kmeans = Kmeans(n_clusters=3, n_jobs=-1, random_state=22)
        mean_shift = MeanShift(bandwidth=50, bin_seeding=True, max_iter=100)
        mean_shift.fit(x)
        print(mean_shift.labels_)

        groups = {}
        for file, cluster in zip(filenames, mean_shift.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)

        for i in list(set(mean_shift.labels_)):
            for j in groups[i]:
                if not os.path.exists(f'../vc_img/gc-{self.geo_cluster}-vc-{i}'):
                    os.mkdir(f'../vc_img/gc-{self.geo_cluster}-vc-{i}')
                shutil.copyfile(f'../photos/{j}', f'../vc_img/gc-{self.geo_cluster}-vc-{i}/{j}')

