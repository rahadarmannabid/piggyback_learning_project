from youtube_transcript_api import YouTubeTranscriptApi
import os
from functools import partial
from multiprocessing.pool import Pool
import cv2
import youtube_dl
import sys
import pafy
import urllib.parse as urlparse

vid_ids = "RsGIDugN4lo"
video = "RsGIDugN4lo"
transcribe  = YouTubeTranscriptApi.get_transcript(video,languages =['en'])
print(transcribe)
transcribe[0]





# #Ask the user for url input
# url = input("Enter Youtube Video URL: ")

# #Getting video id from the url string
# url_data = urlparse.urlparse(url)
# query = urlparse.parse_qs(url_data.query)
# id = query["v"][0]
# video = 'https://youtu.be/{}'.format(str(id))

for i in range(0, len(transcribe)):
        start_time = transcribe[i]['start']
        duration = transcribe[i]['duration']
        end_time = duration + start_time

        #Using the pafy library for youtube videos
        urlPafy = pafy.new(video)
        videoplay = urlPafy.getbest(preftype="any")

        cap = cv2.VideoCapture(videoplay.url)

        #Asking the user for video start time and duration in seconds
        milliseconds = 2000
        # start_time = int(input("Enter Start time: "))
        # end_time = int(input("Enter Length: "))
        # end_time = start_time + end_time

        # Passing the start and end time for CV2
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time*milliseconds)

        #Will execute till the duration specified by the user
        print(transcribe[i])
        while True and cap.get(cv2.CAP_PROP_POS_MSEC)<=end_time*milliseconds:
                success, img = cap.read()
                cv2.imshow("Image", img)
                cv2.waitKey(10)



