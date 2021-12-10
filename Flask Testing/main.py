
### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import cv2
import requests

from feat import Detector

from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


# Flask imports
from flask import Flask, render_template, request, flash, Response


tmdb_api_key = "93e5205a2433162dd3a4b9fdf01ede9f"
tmdb_url = "https://api.themoviedb.org/3/movie/"


def gen():
    """
    Video streaming generator function.
    """
    video_capture = cv2.VideoCapture(0)
    end = 0
    predictions = []

    # Timer
    global k
    k = 0
    max_time = 5
    start = time.time()

    # Record for 15 seconds
    while end - start < max_time:

        k = k + 1
        end = time.time()

        ret, frame = video_capture.read()

        # For flask, save image as t.jpg (rewritten at each step)
        cv2.imwrite('tmp/t.jpg', frame)

        # Yield the image at each step
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('tmp/t.jpg', 'rb').read() + b'\r\n')

        prediction = detector.detect_image("tmp/t.jpg")
        predictions.append(prediction.emotions())

        if end - start > max_time - 1:
            result = pd.concat(predictions, ignore_index=True, names=[
                               'anger',	'disgust',	'fear',	'happiness', 'sadness',	'surprise',	'neutral']).dropna()
            result.to_csv("static/js/db/prob.csv")
            break

    video_capture.release()

# Config other functionality


def emotion_label(emotion):
    if emotion == 0:
        return "Angry"
    elif emotion == 1:
        return "Disgust"
    elif emotion == 2:
        return "Fear"
    elif emotion == 3:
        return "Happy"
    elif emotion == 4:
        return "Sad"
    elif emotion == 5:
        return "Surprise"
    else:
        return "Neutral"


def genre_label(genre_ids):
    genre_list = []
    for id in genre_ids:
        if id == 0:
            genre_list.append("Action")
        elif id == 1:
            genre_list.append("Adventure")
        elif id == 2:
            genre_list.append("Animation")
        elif id == 3:
            genre_list.append("Children's")
        elif id == 4:
            genre_list.append("Comedy")
        elif id == 5:
            genre_list.append("Crime")
        elif id == 6:
            genre_list.append("Documentary")
        elif id == 7:
            genre_list.append("Drama")
        elif id == 8:
            genre_list.append("Fantasy")
        elif id == 9:
            genre_list.append("Film-Noir")
        elif id == 10:
            genre_list.append("Horror")
        elif id == 11:
            genre_list.append("Musical")
        elif id == 12:
            genre_list.append("Mystery")
        elif id == 13:
            genre_list.append("Romance")
        elif id == 14:
            genre_list.append("Sci-Fi")
        elif id == 15:
            genre_list.append("Thriller")
        elif id == 16:
            genre_list.append("War")
        elif id == 17:
            genre_list.append("Western")
    return genre_list


def filter_movie_recommendations_by_emotion(rec_movie_ids, predicted_emotional_state, movie_df):
    emotional_state_dict = {
        "Angry": ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                  'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                  'Western'],
        "Disgust": ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Drama', 'Fantasy', 'Musical',
                    'Mystery', 'Romance', 'Sci-Fi', 'Western'],
        "Fear": ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Documentary', 'Fantasy', 'Musical',
                 'Mystery', 'Romance', 'Sci-Fi', 'Western'],
        "Happy": ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                  'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                  'Western'],
        "Sad": ['Adventure', 'Animation', "Children's", 'Comedy', 'Musical', 'Romance'],
        "Surprise": ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Documentary', 'Fantasy', 'Musical',
                     'Romance', 'Sci-Fi', 'Western'],
        "Neutral": ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                    'Western']}

    rec_movies = movie_df[movie_df["movieId"].isin(rec_movie_ids)]

    filtered_rec_movies = []
    for row in rec_movies.itertuples():
        rec_movie_genres = row.genres.split('|')
        if set(rec_movie_genres).issubset(set(emotional_state_dict[emotion_label(predicted_emotional_state)])):
            filtered_rec_movies.append(row)

    return pd.DataFrame(filtered_rec_movies)


# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'


################################################################################
################################## INDEX #######################################
################################################################################


@app.route('/video', methods=['GET'])
def video():
    global genres
    genres = request.args.get("genres")
    # Display a warning message
    flash(
        'We will detect your emotion in 5 secons. Due to restrictions, we are not able to redirect you once the video is over. Please move your URL to /movie instead of /video once over. You will be able to see your results then.')
    return render_template('video.html')


# Display the video flow (face, landmarks, emotion)
@app.route('/video_1', methods=['POST'])
def video_1():
    try:
        # Response is used to display a flow of information
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except:
        return None


# Dashboard
@app.route('/movie', methods=("POST", "GET"))
def video_dash():
    # Load personal history
    df = pd.read_csv('static/js/db/df.csv')
    movie_df = pd.read_csv('static/js/db/movie_df.csv')

    df_2 = pd.read_csv('static/js/db/prob.csv')
    max_props = np.argmax(df_2.to_numpy(), axis=1)
    _, max_counts = np.unique(max_props, return_counts=True)
    total_counts = [np.count_nonzero(max_props == i) for i in range(7)]
    emotion = np.argmax(total_counts)

    # Save dataframes to the local storage

    new_user_id = df['userId'].max() + 1
    favorite_genres_db = map(int, genres.split(","))
    favorite_genres = genre_label(favorite_genres_db)

    applicable_movie_ids = []
    applicable_movie_names = []
    for row in movie_df.itertuples():
        movie_genres = row.genres.split('|')
        if set(movie_genres).issubset(set(favorite_genres)):
            applicable_movie_names.append(row.title)
            applicable_movie_ids.append(row.movieId)

    added_movie_names = applicable_movie_names[:10]
    added_movie_ids = applicable_movie_ids[:10]

    for idx, new_movie in enumerate(added_movie_names):
        df = df.append({'userId': int(new_user_id), 'movieId': int(added_movie_ids[idx]), 'rating': 5.0,
                        'timestamp': int(964982703)}, ignore_index=True)

    df[['userId', 'movieId', 'timestamp']] = df[[
        'userId', 'movieId', 'timestamp']].astype('int32')

    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df["rating"] = df["rating"].values.astype(np.float32)
    # min and max ratings will be used to normalize the ratings later
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    df = df.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    # Normalize the targets between 0 and 1. Makes it easy to train.
    y = df["rating"].apply(lambda x: (x - min_rating) /
                           (max_rating - min_rating)).values
    # Assuming training on 90% of the data and validating on 10%.
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )

    EMBEDDING_SIZE = 100
    users_input = layers.Input(shape=(1,), name="users_input")
    users_embedding = layers.Embedding(
        num_users + 1, EMBEDDING_SIZE, name="users_embeddings")(users_input)
    users_bias = layers.Embedding(
        num_users + 1, 1, name="users_bias")(users_input)

    movies_input = layers.Input(shape=(1,), name="movies_input")
    movies_embedding = layers.Embedding(
        num_movies + 1, EMBEDDING_SIZE, name="movies_embedding")(movies_input)
    movies_bias = layers.Embedding(
        num_movies + 1, 1, name="movies_bias")(movies_input)

    dot_product_users_movies = layers.multiply(
        [users_embedding, movies_embedding])
    input_terms = dot_product_users_movies + users_bias + movies_bias
    input_terms = layers.Flatten(name="fl_inputs")(input_terms)
    output = layers.Dense(1, activation="relu", name="output")(input_terms)

    model = tf.keras.Model(inputs=[users_input, movies_input], outputs=output)
    opt_adam = tf.keras.optimizers.Adam(lr=0.005)
    model.compile(optimizer=opt_adam, loss=[
                  'mse'], metrics=['mean_absolute_error'])
    model.fit(x=[x_train[:, 0], x_train[:, 1]], y=y_train, batch_size=64, epochs=2, verbose=1,
              validation_data=([x_val[:, 0], x_val[:, 1]], y_val))

    user_id = new_user_id

    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [
        [movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(
        [user_movie_array[:, 0], user_movie_array[:, 1]]).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]

    filtered_recommendations = filter_movie_recommendations_by_emotion(
        recommended_movie_ids, emotion, movie_df)

    # print(filtered_recommendations)
    print("Top movie recommendations")
    print("----" * 8)
    movie_genres = ""
    title = ""

    for row in filtered_recommendations.itertuples():
        movie_id = row.movieId
        movie_genres = row.genres
        title = row.title
        break

    links_df = pd.read_csv('static/js/db/links_df.csv')
    tmdbId = links_df.loc[links_df['movieId'] == movie_id, "tmdbId"].values[0]
    movie_url = tmdb_url + str(int(tmdbId)) + "?api_key=" + tmdb_api_key
    credits_url = tmdb_url + str(int(tmdbId)) + \
        "/credits?api_key=" + tmdb_api_key

    movie_response = requests.get(movie_url).json()
    poster_path = "https://image.tmdb.org/t/p/w500" + \
        movie_response["poster_path"]

    credits_response = requests.get(credits_url).json()
    director = ""

    for member in credits_response["crew"]:

        if member['job'] == "Director":
            director = member['name']
            break

    cast = []

    for member in credits_response["cast"][:3]:
        cast.append(member['name'])

    # Change template in final output
    return render_template('movie.html', title=title, director=director, cast=cast, poster=poster_path, genres=movie_genres)


if __name__ == '__main__':
    face_model = "mtcnn"
    landmark_model = "mobilenet"
    au_model = "rf"
    emotion_model = "resmasknet"
    detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model,
                        emotion_model=emotion_model)

    # Download the actual data from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    # Use the ratings.csv file
    movielens_data_file_url = (
        "http://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    )
    movielens_zipped_file = keras.utils.get_file(
        "ml-latest-small.zip", movielens_data_file_url, extract=False
    )
    keras_datasets_path = Path(movielens_zipped_file).parents[0]
    movielens_dir = keras_datasets_path / "ml-latest-small"

    # Only extract the data the first time the script is run.
    if not movielens_dir.exists():
        with ZipFile(movielens_zipped_file, "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=keras_datasets_path)
            print("Done!")

    ratings_file = movielens_dir / "ratings.csv"
    movie_df = pd.read_csv(movielens_dir / "movies.csv")
    links_df = pd.read_csv(movielens_dir / "links.csv")
    df = pd.read_csv(ratings_file)

    movie_df.to_csv('static/js/db/movie_df.csv')
    links_df.to_csv('static/js/db/links_df.csv')
    df.to_csv('static/js/db/df.csv')

    app.run(host='0.0.0.0', debug=True)
