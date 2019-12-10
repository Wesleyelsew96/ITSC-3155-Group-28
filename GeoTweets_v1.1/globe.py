import pygame
import math
from geopy.geocoders import Nominatim
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL import shaders
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

import requests
import numpy as np
import tweepy
import csv
import json

consumer_key = 'XUcSEg1cvJvD1rcCdQlLlOroi'
consumer_secret = 'vWtWuYdEgCmQeH31vYuAZdNTTJo5ivDMGOfVOczLbc7dJkO21L'

access_token = '1176619957615915008-zkKKA9Piu3u3Z8tahDJIZfNE27DAK4'
access_token_secret = 'tsYvi8GEp0ju3o08UQAwYh8jr5snVaNEjAdtVcXhNP9WR'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

geolocator = Nominatim(user_agent="GEO_Tweet")

VERT_SHADER = """
    #version 330
    
    in vec4 vPosition;
    void main(){
        gl_Position = vPosition;
    }
"""

FRAG_SHADER = """
    #version 330
    
    void main(){
        gl_FragColor = vec4(0,0,1,1);
    }
    
"""

def read_texture(filename):

    """
    Reads an image file and converts to a OpenGL-readable textID format
    """
    img = Image.open(filename)
    img_data = np.array(list(img.getdata()), np.int8)
    textID = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, textID)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                 img.size[0], img.size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
    return textID

def get_user_location():
    send_url = 'http://freegeoip.net/json'
    r = requests.get(send_url)
    j = json.loads(r.text)
    print(j)
    lat = j['latitude']
    lon = j['longitude']
    return [lat,lon]


def get_tweets(query,write,num):
    tweets = []
    if(write.lower() == "y"):
        csvFile = open(query + '.csv', 'a', encoding='utf-8')
        csvWriter = csv.writer(csvFile)
    for tweet in tweepy.Cursor(api.search,q=query,count=num,lang="en").items(num):
        coords = get_coords(tweet)
        print(coords)
        if not coords == []: 
            if(write.lower() == "y"):
                #csvWriter.writerow(["coords","Location","User Name","Text"])
                csvWriter.writerow([coords,tweet.user.location,tweet.created_at,tweet.user.name,tweet.text.encode('utf-8')])
            
            tweets.append([
            coords,
            tweet.user.location,
            tweet.created_at,
            tweet.user.name,
            tweet.text.encode('utf-8')
            ])
    return tweets

def get_tweets_file(path):
    tweets = []
    with open(path + '.csv', newline='',encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            if not row == []:
                #print(row)
                row[0] = row[0].strip(']');
                row[0] = row[0].strip('[');
                
                coords = row[0].split(",")
                coords = [float(coords[0]), float(coords[1])]
                row[0] = coords
                tweets.append(row)
    return tweets
def get_coords(tweet):
        coords = []
        if (tweet.coordinates is not None):
            coords = [tweet.coordinates['coordinates'][0], tweet.coordinates['coordinates'][1]]
            
        elif (tweet.place is not None):
            bbox = tweet.place.bounding_box.coordinates
            coords = [bbox[0][0][0], bbox[0][0][1]]
            
        elif (tweet.user.location is not None and not tweet.user.location == ""):
            location = geolocator.geocode(tweet.user.location)
            if location is not None: 
                coords = [location.latitude, location.longitude]
        return coords

def rotate( angle, axis ):    
    v = normalize( axis )

    x = v[0];
    y = v[1];
    z = v[2];

    c = math.cos( radians(angle) )
    omc = 1.0 - c
    s = math.sin( radians(angle) )

    result =[
            [x*x*omc + c,   x*y*omc - z*s, x*z*omc + y*s, 0.0 ],
            [x*y*omc + z*s, y*y*omc + c,   y*z*omc - x*s, 0.0 ],
            [x*z*omc - y*s, y*z*omc + x*s, z*z*omc + c,   0.0 ],
            [0,0,0,1]
    ]

    return result

def radians( degrees ): 
    return degrees * math.pi / 180.0

def normalize(v):
  length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
  #make sure we don't divide by 0.
  if (length > 0.00001):
    return [v[0] / length, v[1] / length, v[2] / length]
  else:
    return [0, 0, 0]
  
def mult(u,v):
    v = [v[0],v[1],v[2],1]
    result = []
    for r in range(len(u)):
        sum = 0.0
        for c in range(len(u)):
            sum += u[r][c] * v[c]
        result.append(sum)
    
    return [result[0],result[1],result[2]]

def get_tweet_positions(tweets):
    #rotate X then Z 
    #Lat -> Long
    base = [0,-1.01,0,]
    result = []
    
    for tweet in tweets:
        geo = tweet[0]
        v = mult(rotate(geo[0],[1,0,0]),base)
        v = mult(rotate(-geo[1],[0,0,1]),v)
        result.append(v)
        
    return result
      
def statistics():
    return

def main():
    query = input("Enter FilePath. If no File Press Enter: ")
    query = query.rstrip(" ")
    if query == "":
        query = input("Enter a hashtag to find: ")
        query_hash = query
        query = input("Write query to file? y/n?: ")
        query_write = query
        num = int(input("How many tweets? : "))
        tweets = get_tweets(query_hash, query_write, num)
    else:
        tweets = get_tweets_file(query)
        
    #user_location = get_user_location()
    
    tweets_positions = get_tweet_positions(tweets)
    tweets_positions = np.array(tweets_positions, dtype = np.float32)
    
    pygame.init()
    display = (800, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption('PyOpenGLobe')
    pygame.key.set_repeat(1, 10)    # allows press and hold of button

    gluPerspective(40, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)    # sets initial zoom so we can see globe
    
    vertexShader = shaders.compileShader(VERT_SHADER, GL_VERTEX_SHADER)
    fragmentShader = shaders.compileShader(FRAG_SHADER, GL_FRAGMENT_SHADER)
    program = shaders.compileProgram(vertexShader,fragmentShader);
    
    vBuffer = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER,vBuffer)
    glBufferData(GL_ARRAY_BUFFER, tweets_positions, GL_STATIC_DRAW)
    
    vPosition = glGetAttribLocation(program, "vPosition")
    glVertexAttribPointer(vPosition, 3, GL_FLOAT,False,0,None)
    glEnableVertexAttribArray(vPosition)
    
    lastPosX = 0
    lastPosY = 0
    
    R_x = 0
    R_y = 0
    
    texture = read_texture('world.jpg')

    while True:
        for event in pygame.event.get():    # user avtivities are called events

            # Exit cleanly if user quits window
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

            # Rotation with arrow keys
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    glRotatef(1, 0, 1, 0)
                if event.key == pygame.K_RIGHT:
                    glRotatef(1, 0, -1, 0)
                if event.key == pygame.K_UP:
                    glRotatef(1, -1, 0, 0)
                if event.key == pygame.K_DOWN:
                    glRotatef(1, 1, 0, 0)

            # Zoom in and out with mouse wheel
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # wheel rolled up
                    glScaled(1.05, 1.05, 1.05)
                if event.button == 5:  # wheel rolled down
                    glScaled(0.95, 0.95, 0.95)

            # Rotate with mouse click and drag
            if event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                dx = x - lastPosX
                dy = y - lastPosY
                mouseState = pygame.mouse.get_pressed()
                if mouseState[0]:

                    modelView = (GLfloat * 16)()
                    mvm = glGetFloatv(GL_MODELVIEW_MATRIX, modelView)

                    # To combine x-axis and y-axis rotation
                    temp = (GLfloat * 3)()
                    temp[0] = modelView[0]*dy + modelView[1]*dx
                    temp[1] = modelView[4]*dy + modelView[5]*dx
                    temp[2] = modelView[8]*dy + modelView[9]*dx
                    norm_xy = math.sqrt(temp[0]*temp[0] + temp[1]
                                        * temp[1] + temp[2]*temp[2])
                    glRotatef(math.sqrt(dx*dx+dy*dy),
                              temp[0]/norm_xy, temp[1]/norm_xy, temp[2]/norm_xy)

                lastPosX = x
                lastPosY = y

        # Creates Sphere and wraps texture
        glEnable(GL_DEPTH_TEST)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        qobj = gluNewQuadric()
        gluQuadricTexture(qobj, GL_TRUE)
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture)
        gluSphere(qobj, 1, 50, 50)
        gluDeleteQuadric(qobj)
        glDisable(GL_TEXTURE_2D)
        
        glPointSize(5);
        glDrawArrays(GL_POINTS, 0, len(tweets_positions))
        
        # Displays pygame window
        pygame.display.flip()
        pygame.time.wait(10)


main()
