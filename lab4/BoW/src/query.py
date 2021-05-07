
import argparse as ap
import cv2
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *
from PIL import Image


# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-i", "--image", help="Path to query image", required="True")
args = vars(parser.parse_args())

# Get query image path
image_path = args["image"]

# Load the classifier, class names, scaler, number of clusters and vocabulary
im_features, image_paths, idf, numWords, voc = joblib.load("bof.pkl")

# Create feature extraction and keypoint detector objects
fea_det = cv2.ORB_create()
des_ext = cv2.ORB_create()

# List where all the descriptors are stored
des_list = []
im = cv2.imread(image_path)
kpts = fea_det.detect(im)
kpts, des = des_ext.compute(im, kpts)
des_list.append((image_path, des))

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
descriptors = descriptors.astype(float)
test_features = np.zeros((1, numWords), "float32")
words, distance = vq(descriptors,voc)
for w in words:
    test_features[0][w] += 1

# Perform Tf-Idf vectorization and L2 normalization
test_features = test_features*idf
test_features = preprocessing.normalize(test_features, norm='l2')
score = np.dot(test_features, im_features.T)
rank_ID = np.argsort(-score)

# Visualize the results
figure()
gray()
fig=subplot(3,3,1)
imshow(im[:,:,::-1])
title("Query: "+image_path[-9:-4])
axis('off')
for i, ID in enumerate(rank_ID[0][0:6]):
	img = Image.open(image_paths[ID])
	gray()
	subplot(3,3,i+4)
	imshow(img)
	title("Detected:"+str(i+1)+" "+image_paths[ID][-9:-4])
	axis('off')
show()
