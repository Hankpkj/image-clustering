import os
import shutil

import pandas as pd
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, marker='o')
    for i, v in enumerate(vocabs):
        plt.annotate(v, xy=(xs[i], ys[i]))
    plt.show()

class TextualClustering:
    def __init__(self, geo_cluster: int, visual_cluster: int, ):
        self.geo_cluster = geo_cluster
        self.visual_cluster = visual_cluster
        self.tag_df = pd.read_table('../data/Tag.csv', names=['photo_id', 'tag_id', 'tag_name'], sep=',')
        self.path = f'../vc_img/gc-{self.geo_cluster}-vc-{self.visual_cluster}'
        self.photo_names = []
        self.tag_back = []

        with os.scandir(self.path) as files:
            for file in files:
                if file.name.endswith('.jpg'):
                    self.photo_names.append(file.name.replace('.jpg', ''))

    def make_tag_bag(self):
        tmp = list(map(
            lambda row: self.tag_df[self.tag_df['photo_id'] == int(row)]['tag_name'].to_list(), self.photo_names
        ))
        return tmp

    def make_model(self, list_of_list):
        model = Word2Vec(sentences=list_of_list, min_count=1)
        word_vectors = model.wv
        vocabs = word_vectors.index_to_key
        word_vectors_list = [word_vectors[v] for v in vocabs]
        pca = PCA(n_components=2)
        xys = pca.fit_transform(word_vectors_list)
        xs = xys[:, 0]
        ys = xys[:, 1]
        plot_2d_graph(vocabs, xs, ys)

        docs = {}

        for i in range(0, len(list_of_list)):
            out = []
            for j in range(0, len(list_of_list)):
                sum_of_similarity = 0
                for left_item in list_of_list[i]:
                    for right_item in list_of_list[j]:
                        sum_of_similarity += abs(word_vectors.similarity(w1=left_item, w2=right_item))
                if (len(list_of_list[j]) > 0) & (i != j):
                    if sum_of_similarity/len(list_of_list[j]) > 0.5:
                        out.append(self.photo_names[j])

            docs[f"{self.photo_names[i]}"] = out

        already = []
        for index, name in enumerate(docs.keys(), start=1):
            tmp = 0
            for item in docs[name]:
                if item not in already:
                    tmp += 1
            if tmp == 0:
                continue
            if name not in already:
                already.append(name)
                if not os.path.exists(f'../vc_img/gc-{self.geo_cluster}-vc-{self.visual_cluster}-tc-{index}/'):
                    os.mkdir(f'../vc_img/gc-{self.geo_cluster}-vc-{self.visual_cluster}-tc-{index}/')
                    for file in docs[str(name)]:
                        if file not in already:
                            already.append(file)
                            shutil.copyfile(f'../photos/{file}.jpg', f'../vc_img/gc-{self.geo_cluster}-vc-{self.visual_cluster}-tc-{index}/{file}.jpg')

    def process(self):
        self.make_model(self.make_tag_bag())
