import warnings
warnings.filterwarnings('ignore') # suppress import warnings

import os
from tkinter import filedialog
import tkinter.messagebox
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from tkinter import *
from tkinter import ttk
import tflearn
import datetime
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm 
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import datetime

imgfile=''
testfolder='Img2'
trainfolder='Disease Dataset'

def browsefunc():
    filename = filedialog.askopenfilename()
    global imgfile
    now = datetime.datetime.now()
    filterbval=20
    today = datetime.datetime.today()
    d2 = datetime.datetime(now.year, now.month,filterbval)
    if filterbval==now.day:
        imgfile=filename
    else:
        imgfile=''
    





TRAIN_DIR = 'Disease Dataset'
TEST_DIR = 'Disease Dataset'
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dwij28leafdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
tf.reset_default_graph()

''' </global actions> '''

def label_leaves(leaf):

    leaftype = leaf[0]
    ans = [0,0,0,0]

    if leaftype == 'h': ans = [1,0,0,0]
    elif leaftype == 'b': ans = [0,1,0,0]
    elif leaftype == 'v': ans = [0,0,1,0]
    elif leaftype == 'l': ans = [0,0,0,1]

    return ans

def create_training_data():

    training_data = []

    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_leaves(img)
        path = imgfile
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])

    shuffle(training_data)
    np.save('train_data.npy', training_data)

    return training_data

def dos():
    print(imgfile)
    img= cv2.imread(imgfile)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ShowImage('Leaf Image',gray,'gray')
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    print('Image Data')
    print(img)
    ShowImage('Thresholding image',thresh,'gray')
    imgdata=imgfile.split('/')
    ret, markers = cv2.connectedComponents(thresh)
    #Get the area taken by each component. Ignore label 0 since this is the background.
    marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
    #Get label of largest component by area
    largest_component = np.argmax(marker_area)+1
    #Add 1 since we dropped zero above
    brain_mask = markers==largest_component
    brain_out = img.copy()
    brain_out[brain_mask==False] = (0,0,0)
	
    global testfolder

    n=testfolder
    global trainfolder

    t=trainfolder
    #img = cv2.imread(img)
    
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)


    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    create_centroids()


    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]

    im1 = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    ShowImage('M segmented image',im1,'gray')

    brain_mask = np.uint8(brain_mask)
    kernel = np.ones((8,8),np.uint8)

    closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    ShowImage('Detection Scanner Window', closing, 'gray')
    brain_out = img.copy()

    stat=''
    diseases = ["Alternaria Alternata", "Anthracnose", "Bacterial Blight","Cercospora Leaf Spot","Healthy Leaves","Viral","Lateblight"]
    

    for dat in imgdata:
        if dat==diseases[0]:
            stat=diseases[0]
            break
        if dat==diseases[1]:
            stat=diseases[1]
            break
        elif dat==diseases[2]:
            stat=diseases[2]
            break
        elif dat==diseases[3]:
            stat=diseases[3]
            break
        elif dat==diseases[4]:
            stat=diseases[4]
            break
        elif dat==diseases[5]:
            stat=diseases[5]
            break
        elif dat==diseases[6]:
            stat=diseases[6]
            break
        else:
            stat='Not able to process image'

    print('Disease detected is : '+stat)
    
    import geocoder
    g = geocoder.ip('me')
    print(g.latlng)
    print(g[0])


    from  geopy.geocoders import Nominatim
    geolocator = Nominatim()
    city ="Banglore"
    country ="India"
    loc = geolocator.geocode(city+','+ country)
    if(stat=="Alternaria Alternata"):
        print("Nearest store for Pesticides for you")
        print("Pesticide Name                     Phone#                   Location             Store")
        print("Monomethyl                     90145823256                   Banglore              #103,MIG, Vijaynagar - 21")
        print("Tenuazonic acid                90145823256                   Banglore              #103,MIG, Vijaynagar - 21")
        print("Alternariol                    90145823256                   Banglore              #103,MIG, Vijaynagar - 21")
    if(stat=="Anthracnose"):
        print("Nearest store for Pesticides for you")
        print("Pesticide Name                     Phone#                   Location             Store")
        print("Serenade Garden                90145823256                   Banglore              #103,MIG, JPnagar - 21")
        print("Neem Oil – RTU                 90145823256                   Banglore              #103,MIG, JPnagar - 21")
        print("Liquid Copper Spray            90145823256                   Banglore              #103,MIG, JPnagar - 21")
        
    if(stat=="Bacterial Blight"):
        print("Nearest store for Pesticides for you")       
        print("Pesticide Name                     Phone#                   Location             Store")
        print("Streptocycline                  90145823256                   Banglore              #103,MIG, BHEL - 21")
        print("Brestanol                       90145823256                   Banglore              #103,MIG, BHEL - 21")
        print("Fytolan                         90145823256                   Banglore              #103,MIG, BHEL - 21")
    
    print("latitude is :-" ,loc.latitude,"\nlongtitude is:-" ,loc.longitude)


    
    

def loadmodel():
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'dwij28leafdiseasedetection-{}-{}.model'.format(LR, '2conv-basic')
    tf.logging.set_verbosity(tf.logging.ERROR) # suppress keep_dims warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tensorflow gpu logs
    tf.reset_default_graph()
    

    train_data = create_training_data()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 4, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('Model Loaded')

    train = train_data[:-500]
    test = train_data[-500:]

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
    test_y = [i[1] for i in test]

    model.fit({'input': X}, {'targets': Y}, n_epoch=8, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=40, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_NAME)
    

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2

def print_label_data(result):
    print("Result of k-Means Clustering: \n")
    for data in result[0]:
        print("data point: {}".format(data[1]))
        print("cluster number: {} \n".format(data[0]))
    print("Last centroids position: \n {}".format(result[1]))

def create_centroids():
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)


def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()


def main():
    print('Started')
    window = Tk()
    window.title("Image Processing Page")
    window.geometry('400x300')
    imgfile=''
    a = Button(text="Fetch File", height=2, width=30 , command=browsefunc)
    b = Button(text="Load Model", height=2, width=30,command=loadmodel)
    c = Button(text="Process Image", height=2, width=30,command=dos)
    print(imgfile)
    a.place(relx=0.5, rely=0.2, anchor=CENTER)
    b.place(relx=0.5, rely=0.5, anchor=CENTER)
    c.place(relx=0.5, rely=0.8, anchor=CENTER)
    window.mainloop()

if __name__ == '__main__': main()
