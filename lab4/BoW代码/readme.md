运行环境
python2.7
opencv3.4
sklearn
numpy
matplotlib
首先训练模型，语法如下

文件的目录结构:
````
python searchFeatures.py -t 需要训练的图片文件夹路径
#例如python searchFeatures.py -t ../dataset/test/
````

进行查找，本程序只能每次查找一张图片，并返回与之匹配度（递减）最接近的6张图片
````bash
python query.py -i TestImgPath
#例如python query.py -i ../dataset/test/img.ppm
````

