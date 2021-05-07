#-*- coding:utf-8 –*-
import argparse as ap
import cv2
import numpy as np
import os
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from scipy import spatial

parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="训练集路径", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"] 


training_names = os.listdir(train_path)
numWords = 1000 #视觉词的数量

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
for training_name in training_names:
    image_path = os.path.join(train_path, training_name)
    image_paths += [image_path] #得到每张图片的路径加到list中

# Create feature extraction and keypoint detector objects

fea_det = cv2.xfeatures2d.SIFT_create()
des_ext = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []
each_fea_list = []
only_des_list = []
des_to_fea_dict = {}
des_to_img_path_dict = {}

for i, image_path in enumerate(image_paths):
    im = cv2.imread(image_path)
    print ("提取了图片 %s 的SIFT特征, %d of %d images" %(training_names[i], i, len(image_paths)))
    kpts = fea_det.detect(im) #->keypoints
    kpts, des = des_ext.compute(im, kpts) #→ keypoints, descriptors
    if i == 1:
         print len(des)
    for j in range(len(des)):
        des_to_fea_dict[des[j].tobytes()] = kpts[j]
        des_to_img_path_dict[des[j].tobytes()] = image_path #每个特征点和对应的描述子以及图像建立联系，方便回溯
    des_list.append((image_path, des)) #每个元素是图片以及某个特征

print(len(des_list))
for des in des_list: #des_list内含38个tuple
    for each_fea in des[1]: #des[0] = image_path，des[1] = 500*128大小的二维数组
        each_fea_list.append((des[0], each_fea)) #得到128维descriptor
	only_des_list.append(each_fea)
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))


# 执行k-means聚类
print ("开始 k-means 聚类: 目标视觉词个数：%d 个,目前已提取特征点个数：%d个" %(numWords, descriptors.shape[0]))
descriptors = descriptors.astype(float)
codebook, _= kmeans(descriptors, numWords, 1)
#codebook有1000个元素，每个元素是一个128维的聚类中心（SIFT是128维）
print (np.size(codebook[0]))

print len(only_des_list)
tree = spatial.KDTree(only_des_list)
#print (tree.data)

for i, coder in enumerate(codebook):
    mindist = 1e9 + 0.05
    index = ""
    index_fea = np.array((32,1), "int")
#    for j, packer in enumerate(each_fea_list):
#	#print (imger)
#        imger, each_fea = packer
#        dist = np.linalg.norm(each_fea - coder)
#        if dist < mindist:
#            index = j
#            mindist = dist
    local_des = tree.data[tree.query(coder)[1]]
    local_fea = des_to_fea_dict[local_des.tobytes()]
    local_img_path = des_to_img_path_dict[local_des.tobytes()]
    print("code %d 属于 第 %s 张图片" %(i, index))
    if i < 20:
        
	tmp_fea_structure = np.expand_dims(local_fea, axis = 0)
	print(tmp_fea_structure)
	img1 = cv2.imread(local_img_path)
	rows, cols, channel = img1.shape
        #img2 = cv2.drawKeypoints(img1, tmp_fea_structure, outImage=None)
	r = tmp_fea_structure[0].size
	left_up = (int(tmp_fea_structure[0].pt[0] - r), int(tmp_fea_structure[0].pt[1] - r))
	right_down = (int(tmp_fea_structure[0].pt[0] + r), int(tmp_fea_structure[0].pt[1] + r))
	center = (int(tmp_fea_structure[0].pt[0]), int(tmp_fea_structure[0].pt[1]))
	if (left_up[0] < 0) | (left_up[1] < 0):
	    continue;
	if (right_down[0] > img1.shape[0]) | (right_down[1] > img1.shape[1]):
	    continue;
	#img2 = img1[int(tmp_fea_structure[0].pt[0] - r / 2.0):int(tmp_fea_structure[0].pt[0] + r / 2.0), int(tmp_fea_structure[0].pt[1] - r / 2.0):int(tmp_fea_structure[0].pt[1] + r / 2.0)]
	print (tmp_fea_structure[0].size)
        cv2.circle(img1, center, int(r) ,(255,255,0), 3)

	# 创建一张4通道的新图片，包含透明通道，初始化是透明的
	img_new = np.zeros((rows,cols,4),np.uint8)
	img_new[:,:,0:3] = img1[:,:,0:3]
	# 创建一张单通道的图片，设置最大内接圆为不透明，注意圆心的坐标设置，cols是x坐标，rows是y坐标
	img_circle = np.zeros((rows,cols,1),np.uint8)
	img_circle[:,:,:] = 0  # 设置为全透明
	img_circle = cv2.circle(img_circle, center , int(r),(255),-1) # 设置最大内接圆为不透明
	# 图片融合
	img_new[:,:,3] = img_circle[:,:,0]
	img_newer = img_new[left_up[1]:right_down[1],left_up[0]:right_down[0],:]
	# 保存图片
	cv2.imwrite(str(i) + "_.png", img_newer)
	#cv2.rectangle(img2, left_up, right_down, (255,255,0), 3)
        cv2.imwrite(str(i) + ".jpg",img1)

# 计算特征直方图
im_features = np.zeros((len(image_paths), numWords), "float32")
for i in xrange(len(image_paths)):
    words = vq(des_list[i][1],codebook)[0] #使用vq函数根据聚类中心对所有数据进行分类,vq的输出也是两维的,[0]表示的是所有数据的label
    for w in words:
        im_features[i][w] += 1 #对应的视觉词频数+1

# Perform Tf-Idf vectorization TF-IDF（term frequency–inverse document frequency）是一种统计方法，用来衡量字词对于文本的重要程度
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Perform L2 normalization
im_features = im_features*idf
im_features = preprocessing.normalize(im_features, norm='l2')

joblib.dump((im_features, image_paths, idf, numWords, codebook), "bof.pkl", compress=3)

