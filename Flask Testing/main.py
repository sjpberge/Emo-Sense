
### General imports ###
from __future__ import division
import numpy as np
import pandas as pd
import time
import re
import os
from collections import Counter
import altair as alt
import cv2

from feat import Detector
from feat.tests.utils import get_test_data_path



### Flask imports
import requests
from flask import Flask, render_template, session, request, redirect, flash, Response


def gen():
    """
    Video streaming generator function.
    """
    # face_model = "mtcnn"
    # landmark_model = "mobilenet"
    # au_model = "rf"
    # emotion_model = "resmasknet"
    # detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model, emotion_model=emotion_model)

    emotional_labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

    # Start video capute. 0 = Webcam, 1 = Video file, -1 = Webcam for Web
    video_capture = cv2.VideoCapture(0)



    # We have 7 emotions
    nClasses = 7

    # Timer until the end of the recording
    end = 0

    # Prediction vector
    predictions = []

    # Timer
    global k
    k = 0
    max_time = 15
    start = time.time()

    angry_0 = []
    disgust_1 = []
    fear_2 = []
    happy_3 = []
    sad_4 = []
    surprise_5 = []
    neutral_6 = []

    # Record for 45 seconds
    while end - start < max_time:

        k = k + 1
        end = time.time()

        # Capture frame-by-frame the video_capture initiated above
        ret, frame = video_capture.read()

        prediction = detector.detect_image(frame)

        # Most likely emotion
        # prediction_result = np.argmax(prediction.emotions())

        # Append the emotion to the final list
        predictions.append(prediction.emotions())


        # For flask, save image as t.jpg (rewritten at each step)
        cv2.imwrite('tmp/t.jpg', frame)

        # Yield the image at each step
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('tmp/t.jpg', 'rb').read() + b'\r\n')

        # Emotion mapping
        # emotion = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

        # Once reaching the end, write the results to the personal file and to the overall file
        if end - start > max_time - 1:
            # with open("static/js/db/histo_perso.txt", "w") as d:
            #     d.write("density" + '\n')
            #     for val in predictions:
            #         d.write(str(val) + '\n')
            #
            # with open("static/js/db/histo.txt", "a") as d:
            #     for val in predictions:
            #         d.write(str(val) + '\n')

            result = pd.concat(predictions, ignore_index=True, names= ['anger',	'disgust',	'fear',	'happiness',
                                                                       'sadness',	'surprise',	'neutral']).dropna()

            result.to_csv("static/js/db/prob.csv")
            break

    video_capture.release()

# Flask config
app = Flask(__name__)
app.secret_key = b'(\xee\x00\xd4\xce"\xcf\xe8@\r\xde\xfc\xbdJ\x08W'
app.config['UPLOAD_FOLDER'] = '/Upload'

################################################################################
################################## INDEX #######################################
################################################################################

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

################################################################################
################################## RULES #######################################
################################################################################

# Rules of the game
@app.route('/rules')
def rules():
    return render_template('rules.html')



################################################################################
############################### VIDEO INTERVIEW ################################
################################################################################

# Read the overall dataframe before the user starts to add his own data
# df = pd.read_csv('static/js/db/histo.txt', sep=",")


# Video interview template
@app.route('/video', methods=['POST'])
def video():
    # Display a warning message
    flash(
        'You will have 45 seconds to discuss the topic mentioned above. Due to restrictions, we are not able to redirect you once the video is over. Please move your URL to /video_dash instead of /video_1 once over. You will be able to see your results then.')
    return render_template('video.html')


# Display the video flow (face, landmarks, emotion)
@app.route('/video_1', methods=['POST'])
def video_1():
    try:
        # Response is used to display a flow of information
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(stream_template('video.html', gen()))
    except:
        return None


# Dashboard
@app.route('/video_dash', methods=("POST", "GET"))
def video_dash():
    # Load personal history
    df_2 = pd.read_csv('static/js/db/prob.csv')
    max_props = np.argmax(df_2.to_numpy(), axis=1)
    _, max_counts = np.unique(max_props, return_counts=True)

    def emo_prop(df_2):
        return [int(100 * np.count_nonzero(max_props == 0) / len(df_2)),
                int(100 * np.count_nonzero(max_props == 1) / len(df_2)),
                int(100 * np.count_nonzero(max_props == 2) / len(df_2)),
                int(100 * np.count_nonzero(max_props == 3) / len(df_2)),
                int(100 * np.count_nonzero(max_props == 4) / len(df_2)),
                int(100 * np.count_nonzero(max_props == 5) / len(df_2)),
                int(100 * np.count_nonzero(max_props == 6) / len(df_2))]

    emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    # emo_perso = {}
    # emo_glob = {}
    #
    # for i in range(len(emotions)):
    #     emo_perso[emotions[i]] = len(df_2[df_2.density == i])
    #     emo_glob[emotions[i]] = len(df[df.density == i])
    #
    # df_perso = pd.DataFrame.from_dict(emo_perso, orient='index')
    # df_perso = df_perso.reset_index()
    # df_perso.columns = ['EMOTION', 'VALUE']
    # df_perso.to_csv('static/js/db/hist_vid_perso.txt', sep=",", index=False)
    #
    # df_glob = pd.DataFrame.from_dict(emo_glob, orient='index')
    # df_glob = df_glob.reset_index()
    # df_glob.columns = ['EMOTION', 'VALUE']
    # df_glob.to_csv('static/js/db/hist_vid_glob.txt', sep=",", index=False)

    emotion = np.argmax(max_counts)
    # emotion_other = df.density.mode()[0]

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

    ### Altair Plot
    df_altair = pd.read_csv('static/js/db/prob.csv', header=None, index_col=None).reset_index()
    df_altair.columns = ['Time', 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    angry = alt.Chart(df_altair).mark_line(color='orange', strokeWidth=2).encode(
        x='Time:Q',
        y='Angry:Q',
        tooltip=["Angry"]
    )

    disgust = alt.Chart(df_altair).mark_line(color='red', strokeWidth=2).encode(
        x='Time:Q',
        y='Disgust:Q',
        tooltip=["Disgust"])

    fear = alt.Chart(df_altair).mark_line(color='green', strokeWidth=2).encode(
        x='Time:Q',
        y='Fear:Q',
        tooltip=["Fear"])

    happy = alt.Chart(df_altair).mark_line(color='blue', strokeWidth=2).encode(
        x='Time:Q',
        y='Happy:Q',
        tooltip=["Happy"])

    sad = alt.Chart(df_altair).mark_line(color='black', strokeWidth=2).encode(
        x='Time:Q',
        y='Sad:Q',
        tooltip=["Sad"])

    surprise = alt.Chart(df_altair).mark_line(color='pink', strokeWidth=2).encode(
        x='Time:Q',
        y='Surprise:Q',
        tooltip=["Surprise"])

    neutral = alt.Chart(df_altair).mark_line(color='brown', strokeWidth=2).encode(
        x='Time:Q',
        y='Neutral:Q',
        tooltip=["Neutral"])

    chart = (angry + disgust + fear + happy + sad + surprise + neutral).properties(
        width=1000, height=400, title='Probability of each emotion over time')

    chart.save('static/CSS/chart.html')

    return render_template('video_dash.html', emo=emotion_label(emotion), emo_other=emotion_label(emotion),
                           prob=emo_prop(df_2), prob_other=emo_prop(df_2))


if __name__ == '__main__':
    face_model = "mtcnn"
    landmark_model = "mobilenet"
    au_model = "rf"
    emotion_model = "resmasknet"
    detector = Detector(face_model=face_model, landmark_model=landmark_model, au_model=au_model,
                        emotion_model=emotion_model)
    app.run(debug=True)
