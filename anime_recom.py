import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from threading import Thread
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans


class KMeans_Anime(Thread):
    def __init__(self):
        self.df1 = pd.read_csv("./archive_anime/anime.csv")
        Thread.__init__(self)
        self.start()

    def run(self):
        print(self.df1.info())
        print(self.df1.head())
        self.anime_rec = df1.drop(
            ["anime_id", "name", "type", "episodes"], axis=1)
        self.anime_rec = self.anime_rec.dropna()
        print(self.anime_rec.info())

        # Label Encoding
        le = LabelEncoder()
        self.anime_rec["genre"] = le.fit_transform(self.anime_rec["genre"])
        print(self.anime_rec.head())
        print(self.anime_rec.corr())
        print(self.anime_rec.describe())

        # Model Creation
        x = np.array(self.anime_rec.iloc[:, :-1])
        wcss = []
        for i in range(1, 11):
            kms = KMeans(n_clusters=i, random_state=42)
            kms.fit(x)
            wcss.append(kms.inertia_)
        plt.plot(range(1, 11), wcss)
        plt.title("The Elbow Method")
        plt.ylabel("Rating")
        plt.xlabel("Genre")
        kms = KMeans(n_clusters=5, random_state=42)
        ypred = kms.fit_predict(x)
        print(ypred)

        # Visualization
        plt.scatter(x[ypred == 0, 0], x[ypred == 0, 1],
                    s=100, c="red", label="Cat 1")
        plt.scatter(x[ypred == 1, 0], x[ypred == 1, 1],
                    s=100, c="green", label="Cat 2")
        plt.scatter(x[ypred == 2, 0], x[ypred == 2, 1],
                    s=50, c="blue", label="Cat 3")
        plt.scatter(x[ypred == 3, 0], x[ypred == 3, 1],
                    s=50, c="yellow", label="Cat 4")
        plt.scatter(x[ypred == 4, 0], x[ypred == 4, 1],
                    s=50, c="orange", label="Cat 5")
        plt.scatter(kms.cluster_centers_[:, 0], kms.cluster_centers_[
                    :, 1], s=100, c="magenta", label="Centroids")
        plt.title("Clusters of Animes")
        plt.xlabel("Rating")
        plt.ylabel("Genre")
        plt.legend()

    def predict(self, name):
        try:
            lst = []
            for item in self.df1["name"]:
                if name == str(item):
                    value = self.df1[self.df1["name"] == name]
                    test_data = value.drop(
                        ["anime_id", "name", "type", "episodes"], axis=1)
            test_data["genre"] = le.transform(test_data["genre"])
            print(test_data)
            x = np.array(test_data.iloc[:, :-1])
            pred = kms.predict(x)
            test_data["genre"] = le.inverse_transform(pred)
            print(test_data, test_data["genre"])
            result_genre = list(np.array(test_data.iloc[:, 0]))
            print(type(result_genre), result_genre)
            return result_genre
        except Exception as err:
            return err


if __name__ == "__main__":
    kmm = KMeans_Anime()
    rg = kmm.predict("Fullmetal Alchemist: Brotherhood")
    try:
        resultant_df = kmm.df1[kmm.df1["listed_in"] == rg[0]]
        print(resultant_df["title"])
    except Exception as err:
        print("Name not found in the database.")
