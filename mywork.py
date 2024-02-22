from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from collections import Counter
import shutil
import socket
import pymysql
from datetime import datetime,time,timedelta
import requests

def diemdanh(masv,malophp,time):
    myconn = pymysql.connect(host = "localhost", user = "root",passwd = "", database = "diemdanh")
    #tạo đối tượng cursor
    cur = myconn.cursor()
    sql = ("INSERT INTO diemdanh.diemdanhtime (masv,malophp,timediemdanh) VALUES (%s,%s,%s);")
    
    #giá trị của một row được cung cấp dưới dạng tuple
    val = (masv,malophp,time)
    try:
        #inserting the values into the table
        cur.execute(sql,val)
    
        #commit the transaction
        myconn.commit()

    except:
        myconn.rollback()
    finally:
        myconn.close()
    print(cur.rowcount,"record inserted!")

def diemdanh2(masv,timediemdanh):
    myconn = pymysql.connect(host="localhost", user="root", passwd="", database="diemdanh")
# Query for creating table
    #now = datetime.now().time()
    current_time = datetime.strptime(timediemdanh,'%Y-%m-%d %H:%M:%S')
#print(current_time)
    mycursor = myconn.cursor()

    mycursor.execute("SELECT malophp,giovaolop FROM lophp")
    alo = timedelta(hours = current_time.hour,minutes = current_time.minute,seconds = current_time.second)
    val = mycursor.fetchall()
#val2 = dict(map(reversed,val))
    maloph = 0
    for i in range(len(val)):
    #print(val[i][1] - timedelta(minutes=30))
        if  (alo < (val[i][1] + timedelta(hours=3))) and (alo > (val[i][1] - timedelta(hours=2))):
            print("ok")
            print(val[i][0])
            maloph = val[i][0]
            break
    querya = ("INSERT INTO diemdanh.diemdanhtime (masv,malophp,timediemdanh) VALUES (%s,%s,%s);")
    valxx = (masv,maloph,timediemdanh)
    try:
    #inserting the values into the table
        mycursor.execute(querya,valxx)
    
        #commit the transaction
        myconn.commit()

    except:
        myconn.rollback()
    finally:
        myconn.close()

def main():
    # s = socket.socket()
    # port = 12345
    # s.bind(('', port))
    # s.listen(5)
    # c, addr = s.accept()
    # Cai dat cac tham so can thiet

    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FOLDER_PATH = 'dataFromClient'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load model da train de nhan dien khuon mat - thuc chat la classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model MTCNN phat hien khuon mat
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lay tensor input va output
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Cai dat cac mang con
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()


            rs = np.array([])
            count = 1
            while (True):
                # if (c.recv(1024).decode() == "diemdanh"):
                #     now = datetime.now()
                #     current_time = now.strftime("%H:%M:%S %d/%m/%Y")
                #     diemdanh(str(new[0][0]),current_time)
                #     #print(cur.rowcount,"record inserted!")
                #     continue
                if len(os.listdir(r'/home/ngoc123sk/Documents/Ky1_21_22/DA_VDK/DoAn/dataFromClient') ) == 0:
                    count = 1
                    continue
                elif len(os.listdir(r'/home/ngoc123sk/Documents/Ky1_21_22/DA_VDK/DoAn/dataFromClient') ) > 19:
                    if count>20:
                        shutil.rmtree(r'/home/ngoc123sk/Documents/Ky1_21_22/DA_VDK/DoAn/dataFromClient')
                        newpath = r'/home/ngoc123sk/Documents/Ky1_21_22/DA_VDK/DoAn/dataFromClient' 
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)
                        count = 1
                        new = Counter(rs.flat).most_common(1)
                        print('da nhan dang duoc sinh vien co ma so sinh vien: '+new[0][0])
                        
                        name = str(new[0][0])
                        res = requests.post('http://192.168.0.100:5000/api', json={"dataFromServer":["MSSV",name]})                      

                        while(True):
                            f = open("example.txt", "r")
                            if (f.read()==""):
                                continue
                            else:
                                f.close()
                                break
                        f = open("example.txt", "r")
                        command = f.readline()
                        name = f.readline()
                        current_time = f.readline()
                        f.close()
                        if (command=='diemdanh\n' or command=='diemdanh'):
                            print('da nhan dc lenh diem danh')
                            #now = datetime.now()
                            #current_time_modifier = datetime.strptime(current_time,"%Y-%m-%d %H:%M:%S")
                            diemdanh2(name,current_time)
                            rs = np.array([])
                            f = open("example.txt", "w")
                            f.write("")
                            f.close()
                            #s.close
                            continue
                        elif(command == "diemdanhlai\n" or command=='diemdanhlai'):
                            print('da nhan dc lenh diem danh lai')
                            rs = np.array([])
                            f = open("example.txt", "w")
                            f.write("")
                            f.close()
                            continue
                        f = open("example.txt", "w")
                        f.write("")
                        f.close()
                        rs = np.array([])
                        #s.close
                        continue
                    cap = cv2.imread(FOLDER_PATH+'/'+str(count)+'.jpg')
                    # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
                    bounding_boxes, _ = align.detect_face.detect_face(cap, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                    faces_found = bounding_boxes.shape[0]
                    try:
                        # Neu co it nhat 1 khuon mat trong cap
                        if faces_found > 0:
                            det = bounding_boxes[:, 0:4]
                            bb = np.zeros((faces_found, 4), dtype=np.int32)
                            for i in range(faces_found):
                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

                                # Cat phan khuon mat tim duoc
                                cropped = cap[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC)
                                scaled = facenet.prewhiten(scaled)
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                                
                                # Dua vao model de classifier
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[
                                    np.arange(len(best_class_indices)), best_class_indices]
                                
                                # Lay ra ten va ty le % cua class co ty le cao nhat
                                best_name = class_names[best_class_indices[0]]
                                
                                print("MSSV: {}, Probability: {}".format(best_name, best_class_probabilities))
                                

                                # Neu ty le nhan dang > 0.8 thi hien thi ten
                                if best_class_probabilities > 0.5:
                                    name = class_names[best_class_indices[0]]
                                    rs = np.append(rs,name)
                                else:
                                    # Con neu <=0.8thi hien thi Unknown
                                    name = "Unknown"
                                    rs = np.append(rs,name)

                                person_detected[best_name] += 1
                    except:
                        pass
                    count= count+1

            
            #cap.release()
            cv2.destroyAllWindows()



main()
