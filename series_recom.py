import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from threading import Thread
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans 

class KMeans_Series(Thread):
    def __init(self):
        self.df1 = pd.read_csv("./archive/netflix_titles.csv")
        Thread.__init__(self)
        self.start()

    def run(self):
        print(self.df1.info())
        print(self.df1.head())
        self.df2=self.df1[self.df1["type"]=="TV Show"]
        print(self.df2["rating"].unique())
        count=1
        for item in self.df2["rating"].unique():
            self.df2["rating"].replace(to_replace=item,value=count,inplace=True)
            count+=1
        self.movie_rec=self.df2.drop(["show_id","type","title","director","cast","country","date_added","duration","description","Unnamed: 12"],axis=1)
        print(self.df2["rating"].unique())
        print(self.movie_rec.info())
        
        # Label Encoding
        le=LabelEncoder()
        self.movie_rec["listed_in"]=le.fit_transform(self.movie_rec["listed_in"])
        print(self.movie_rec.head())
        print(self.movie_rec.corr())
        print(self.movie_rec.describe())

        # Model Creation
        x = np.array(self.movie_rec.iloc[:,1:])
        wcss = [] 
        for i in range(1, 11): 
            kms = KMeans(n_clusters=i, random_state=42) 
            kms.fit(x) 
            wcss.append(kms.inertia_) 
        plt.plot(range(1, 11), wcss) 
        plt.title("The Elbow Method") 
        plt.xlabel("Rating") 
        plt.ylabel("Genre")
        km = KMeans(n_clusters=4,random_state=42)
        ypred = km.fit_predict(x)
        print(ypred)

        # Visualization
        plt.scatter(x[ypred == 0, 0], x[ypred == 0, 1], s=100, c="red", label="Cat 1") 
        plt.scatter(x[ypred == 1, 0], x[ypred == 1, 1], s=100, c="green", label="Cat 2") 
        plt.scatter(x[ypred == 2, 0], x[ypred == 2, 1], s=50, c="blue", label="Cat 3") 
        plt.scatter(x[ypred == 3, 0], x[ypred == 3, 1], s=50, c="yellow", label="Cat 4") 
        plt.title("Clusters of Movies") 
        plt.xlabel("Rating") 
        plt.ylabel("Genre") 
        plt.legend()

    def predict(self,name):
        lst=[]
        for item in self.df2["title"]:
            if name==str(item):
                value = self.df2[self.df2["title"]==name]
                break

        test_data=value.drop(["show_id","type","title","director","cast","country","date_added","duration","description","Unnamed: 12"],axis=1)
        test_data["listed_in"]=le.transform(test_data["listed_in"])
        print(test_data)        
        x = np.array(test_data.iloc[:,1:])            
        pred=kms.predict(x)        
        test_data["listed_in"]=le.inverse_transform(pred)
        print(test_data,test_data["listed_in"])
        result_genre=list(np.array(test_data.iloc[:,2]))
        print(type(result_genre),result_genre)
        return result_genre

if __name__ == "__main__":
    kmm=KMeans_Series()
    rg=kmm.predict("Iverson")
    resultant_df=kmm.df1[kmm.df1["listed_in"]==rg[0]]
    print(resultant_df["title"])
