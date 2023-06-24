from numpy import require
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, url_for, redirect, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/similar-movies/<movie_name>')
def recommender(movie_name):
   try:
      movie_list = recommend(movieId= ds.loc[ds['title'] == movie_name , 'movieId'].iloc[0] , num=5)
      return render_template('Recommendations.html', data=movie_list)
   except:
      return render_template('Error.html', name = movie_name)

@app.route('/home', methods =["GET", "POST"])
def home():
   if request.method == "POST":
      user_movie = request.form['moviename']
      return redirect(url_for('recommender', movie_name = user_movie))
   else:
      user_movie = request.args.get('moviename')
      return redirect(url_for('recommender', movie_name = user_movie))

#ds = pd.read_csv("/home/sayanibiswas/recommender-app/movies.csv")
ds = pd.read_csv(os.path.join(app.root_path, "movies.csv"))

tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(ds['genres'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

results = {}

for idx, row in ds.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], ds['movieId'][i]) for i in similar_indices]

    results[row['movieId']] = similar_items[1:]


def item(id):
    return ds.loc[ds['movieId'] == id]['title'].tolist()[0]


# Just reads the results out of the dictionary.
def recommend(movieId, num):
   recommendations = []
   recs = results[movieId][:num]
   for rec in recs:
      recommendations.append(item(rec[1]))
   return recommendations


if __name__=='__main__':
   app.run(debug= True)