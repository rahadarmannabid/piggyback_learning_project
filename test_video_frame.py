#!/usr/bin/env python
# coding: utf-8

# In[2]:


from youtube_transcript_api import YouTubeTranscriptApi
import os
from functools import partial
from multiprocessing.pool import Pool
import cv2
import youtube_dl
import sys
import pafy
import urllib.parse as urlparse


# In[18]:


transcribe  = YouTubeTranscriptApi.get_transcript("FB0Ym72U6uQ",languages =['en'])
len(transcribe)


# In[24]:



for i in range(0, len(transcribe)):
    print(transcribe[i]["text"], '\n', transcribe[i]["start"]/60 , '\n', transcribe[i]["start"]/60 + transcribe[i]["duration"]/60 )
    


# In[ ]:





# In[25]:


video = "MM97gKF6pGo"


# In[31]:


import os
from functools import partial
from multiprocessing.pool import Pool

import cv2
import youtube_dl


# In[32]:


def process_video_parallel(url, skip_frames, process_number):
    cap = cv2.VideoCapture(url)
    num_processes = os.cpu_count()
    frames_per_process = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // num_processes
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames_per_process * process_number)
    x = 0
    count = 0
    while x < 10 and count < frames_per_process:
        ret, frame = cap.read()
        if not ret:
            break
        filename =r"PATH\shot"+str(x)+".png"
        x += 1
        cv2.imwrite(filename.format(count), frame)
        count += skip_frames  # Skip 300 frames i.e. 10 seconds for 30 fps
        cap.set(1, count)
    cap.release()



video_url = "https://www.youtube.com/watch?v=FB0Ym72U6uQ"  # The Youtube URL
ydl_opts = {}
ydl = youtube_dl.YoutubeDL(ydl_opts)
info_dict = ydl.extract_info(video_url, download=False)

formats = info_dict.get('formats', None)


# In[ ]:




