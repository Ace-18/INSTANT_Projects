import tensorflow as tf
import cv2
import numpy as np
import time,os

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        pass
    
    def readClass(self,classesFilePath):
        with open(classesFilePath,'r') as f:
            self.classesList=f.read().splitlines()
        
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3))
        
        print('Done.')
        
        
   
    def downloadModel(self,URL):
        file=os.path.basename(URL)
        self.modelName=file[:file.index('.')]
        
        self.cacheDir='./pretrained_models'
        os.makedirs(self.cacheDir,exist_ok=True)
        
        get_file(fname=file, origin=URL, cache_dir=self.cacheDir, cache_subdir='checkpoints', extract=True)
       

    
    def load_model(self):
        tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir,'checkpoints',self.modelName,'saved_model'))
        print('model loaded')
        
        
        
        
    def predict(self,image,threshold=0.55):
        
        inputTensor=cv2.cvtColor(image.copy(),cv2.cv2.COLOR_BGR2RGB)
        inputTensor=tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor= inputTensor[tf.newaxis,...]
        detections=self.model(inputTensor)

        bboxes=detections['detection_boxes'][0].numpy()
        classIndexes=detections['detection_classes'][0].numpy().astype(np.int32)
        classScores=detections['detection_scores'][0].numpy()

        h,w,ch=image.shape
        
        bboxIdx= tf.image.non_max_suppression(bboxes, classScores, max_output_size=50, iou_threshold=0.5, score_threshold=threshold)

        if len(bboxIdx)!=0:
            for i in bboxIdx:
                bbox=tuple(bboxes[i].tolist())
                classConf= round(100*classScores[i])
                classIndex= classIndexes[i]

                classLabelText= self.classesList[classIndex]
                classcolor= self.colorList[classIndex]
                
                displayText=f'{classLabelText}: {classConf}'
                ymin, xmin, ymax, xmax= bbox
                ymin, xmin, ymax, xmax= int(ymin*h), int(xmin*w), int(ymax*h), int(xmax*w)
                cv2.rectangle(image,(xmin,ymin),(xmax,ymax),color=classcolor,thickness=1)
                cv2.putText(image,displayText,(xmin,ymin-5),cv2.FONT_HERSHEY_PLAIN,1,classcolor,2)
                
        return image
    
    def predict_img(self,imagePath):
        img=cv2.imread(imagePath)
        
        img=self.predict(img)
        
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow('Image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
    

        





