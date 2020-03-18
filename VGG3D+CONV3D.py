"""
This file contains two models, i.e.VGG3D (by default) and original MicroExpSTCNN. 
"""

import os
import cv2
import numpy
import imageio
import pandas as pd
import xlrd
# from keras import backend as K
from tensorflow.keras import backend as K

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution3D, MaxPooling3D
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils, generic_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import preprocessing

from keras.layers import Conv3D,MaxPooling3D,Flatten
from keras.layers import Dense, Flatten
from keras.models import Sequential

# K.set_image_dim_ordering('th')  # Keras 2.0.0 Strictly
K.image_data_format()

image_rows, image_columns, image_depth = 64, 64, 96


# def get_vid_per_subject(table, listOfIgnoredLabels):
def get_vid_per_subject(table):
    pdt = pd.DataFrame(data=table[0:, 0:], columns=['sub', 'id', 'y'])
    # pdt = pdt[~(pdt['y'].isin(listOfIgnoredLabels))]
    out = pdt.groupby('sub').size().tolist()
    return out


def readinput(path):
    inputList = sorted([f for f in os.listdir(path)])
    # print(len(inputList))

    maxLength = max(len(infile) for infile in inputList)
    minLength = min(len(infile) for infile in inputList)
    # print("inputList:", inputList)
    # print("maxLength {: }".format(maxLength))
    # print("minLength {: }".format(minLength))
    if maxLength == minLength:
        seqList = [path + infile for infile in inputList]

    else:
        # special designed condition for casme2 dB, SMIC should not be involved
        tempList = []
        for index in inputList:
            if len(index) == 12:
                tempVidName = int(index[-5:-4])
            elif len(index) == 13:
                tempVidName = int(index[-6:-4])
            elif len(index) == 14:
                tempVidName = int(index[-7:-4])
            else:
                print("Exceed the predefined range!")
            tempList.append(tempVidName)
        tempList = sorted(tempList)  # reverse = False 升序（默认）
        seqList = [path + 'reg_img' + str(infile) + '.jpg' for infile in tempList]

    return seqList


def label_matching(lable_file, subjects, VidPerSubject):
    label = numpy.loadtxt(lable_file)
    labelperSub = []
    counter = 0
    for sub in range(subjects):
        # print(sub)
        numVid = VidPerSubject[sub]
        labelperSub.append(label[counter:counter + numVid])
        # print("labelperSub:",labelperSub)
        counter = counter + numVid
        # print("labelperSub:-----------------------------",labelperSub)
    print("len(labelperSub):",len(labelperSub))
    # print(labelperSub[1])
    return labelperSub


# loading *label* from CASME2-ObjectiveClasses.xlsx，已经是数字形式
def loading_casme_table(xcel_path):
    wb = xlrd.open_workbook(xcel_path)
    ws = wb.sheet_by_index(0)
    colm = ws.col_slice(colx=0, start_rowx=1, end_rowx=None)
    iD = [str(x.value) for x in colm]
    colm = ws.col_slice(colx=1, start_rowx=1, end_rowx=None)
    vidName = [str(x.value) for x in colm]
    # colm = ws.col_slice(colx=6, start_rowx=1, end_rowx=None)
    colm = ws.col_slice(colx=2, start_rowx=1, end_rowx=None)
    # expression = [str(x.value) for x in colm]
    expression = [int(x.value) for x in colm]
    table = numpy.transpose(numpy.array([numpy.array(iD), numpy.array(vidName), numpy.array(expression)], dtype=str))
    # print(table)
    return table


label_file = "/home/michelexie/桌面/3D-micro-expression-recognition/CASME2-ObjectiveClasses.xlsx"  # classes=7
table = loading_casme_table(label_file)


# print(table.shape)  # (255,3)
rows, columns = table.shape
workplace = "/home/michelexie/桌面/3D-micro-expression-recognition/"
# Loading *images*
inputDir = "/home/michelexie/桌面/3D-micro-expression-recognition/CASME_CROPPED/CASME_CROPPED/"


SubperdB2 = []
n_exp = 7

def read_images(inputDir, data_name=inputDir+'data.npy'):
    if os.path.isfile(data_name):
        print("File exists")
        SubperdB_list=numpy.load(data_name,allow_pickle=True)
    else:
        print("File not exist. Starting generating~~")
        SubperdB_list = []
        SubperdB1 = []
        for sub in sorted([infile for infile in os.listdir(inputDir)]):

            for vid in sorted([inrfile for inrfile in os.listdir(inputDir + sub)]):
                print("\nvid is:", vid)#EP13_06f
                path = inputDir + sub + '/' + vid + '/'  # single image loading path

                print("path:", path)
                imgList = os.listdir(path)
                imgList = readinput(path)

                print("imgList:", imgList)
                numFrame = len(imgList)
                print("numFrame:", numFrame)
                vid_id = numpy.empty([0])

                # FrameperVid1= numpy.empty([0,1, 224, 224, 3])
                FrameperVid1=[]
                for var in range(numFrame):
                    img = cv2.imread(imgList[var])

                    img = cv2.resize(img, (224, 224))
                    FrameperVid1.append(img)
                SubperdB1.append(FrameperVid1)

            SubperdB_list.append(SubperdB1)
        numpy.save(data_name,SubperdB_list)
        print("saving data.npy successfully")
    return SubperdB_list

SubperdB_list = read_images(inputDir)
print("total len(SubperdB_list)",len(SubperdB_list))

label_file_ = "/home/michelexie/桌面/3D-micro-expression-recognition/new_label_.txt"
VidPerSubject = get_vid_per_subject(table)
# print(VidPerSubject)  # [9,13,7,5,9,3...]
subjects = 26
labelperSub = label_matching(label_file_, subjects, VidPerSubject)


def VGG_16():
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1',
                     dim_ordering='tf',
                     input_shape=(224, 224, 3, 24)))
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2'))
    model.add(MaxPooling3D(
        pool_size=(2, 2, 2),
        strides=(2, 2, 2),
        name='block1_pool',
        padding='same'
    ))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2'))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2),  # padding='same',
                           name='block2_pool'
                           ))

    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block3_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block4_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2'))

    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))

    model.add(Dense(4096, activation='relu', name='fc2'))

    model.add(Dense(7, activation='sigmoid', name='predictions'))
    print(model.summary())

    return model
def standard_data_loader(SubjectPerDatabase, y_labels, subjects, classes):
    Train_X = []
    Train_Y = []
    Test_Y_gt = numpy.empty([0])
    for subject in range((subjects)):
        Train_X.append(SubjectPerDatabase[subject])
        Train_Y.append(y_labels[subject])
        Test_Y_gt = numpy.append(Test_Y_gt, y_labels[subject])
    # print(Train_Y)

    # print(Test_Y_gt)
    ############ Conversion to numpy and stacking ###############
    Train_X = numpy.vstack(Train_X)
    Train_Y = numpy.hstack(Train_Y)
    Train_Y = np_utils.to_categorical(Train_Y, classes)
    #############################################################
    # print ("Train_X_shape: " + str(np.shape(Train_X)))
    # print ("Train_Y_shape: " + str(np.shape(Train_Y)))

    return Train_X, Train_Y, Test_Y_gt

def data_loader_with_LOSO(subject, SubjectPerDatabase, y_labels, subjects, classes):
    Train_X = []
    Train_Y = []
    # Real_train_x=[]
    Real_train_x=numpy.empty([0,24,224,224,3])
    Real_test_x = numpy.empty([0, 24, 224, 224, 3])
    num_lists=len(y_labels[subject])
    slice_j1 = numpy.empty([0, 24, 224, 224, 3])
    for l in range(num_lists):
        Test_X = numpy.array(SubjectPerDatabase[subject][l])
        # print("Original Test_X.shape:",Test_X.shape)
        # *slicing array*
        slice_j = Test_X[0:24, :, :, :]
        # print("slice:",slice_j.shape )
        slice_j = numpy.reshape(slice_j, [-1, 24, 224, 224, 3])
        # print("slice_j.shape:",slice_j.shape)
        slice_j1 = numpy.concatenate((slice_j, slice_j1), axis=0)
    Real_test_x = numpy.concatenate((Real_test_x, slice_j1), axis=0)
    print("Real_test_x.shape:",Real_test_x.shape)
    Test_Y = np_utils.to_categorical(y_labels[subject], classes)
    print("Test_Y.shape:",Test_Y.shape)


    ########### Leave-One-Subject-Out ###############
    if subject == 0:
        for i in range(1, subjects):
            print("This is {}th subjects".format(i))
            SubjectPerDatabase=numpy.array(SubjectPerDatabase)
            n=len(y_labels[i])
            Train_X.append(SubjectPerDatabase[i])
            num_pre = 0
            for num1 in range(i):
                num_pre += len(y_labels[num1])
            # print("num_pre:",num_pre)
            slice_j1=numpy.empty([0,24,224,224,3])
            for j in range(num_pre, num_pre+n):
                Train_X[0][j]=numpy.array(Train_X[0][j])
                # print("Train_X[0][j].shape:",Train_X[0][j].shape)
                # *slicing array*
                slice_j=Train_X[0][j][0:24,:,:,:]
                # print("slice:",slice_j.shape )
                slice_j=numpy.reshape(slice_j,[-1,24,224,224,3])
                # print("slice_j.shape:",slice_j.shape)
                slice_j1=numpy.concatenate((slice_j,slice_j1),axis=0)
                # print("slice_j1.shape:",slice_j1.shape)
            Real_train_x=numpy.concatenate((Real_train_x,slice_j1),axis=0)
            Train_Y.append(y_labels[i])

        Train_Y_final = []
        for m in range(25):
            num = len(Train_Y[m])
            for n in range(num):
                # for n in range(24):
                # Train_X_final.append(Train_X[m][n])
                # Train_X_final=numpy.concatenate([Train_X_final,Train_X[m][n]],axis=0)
                Train_Y_final.append(Train_Y[m][n])
        Train_Y = numpy.array(Train_Y_final)
        Train_Y = np_utils.to_categorical(Train_Y, n_exp)

        print("----------------")

    elif subject == subjects - 1:
        for i in range(subjects - 1):
            print("This is {}th subjects".format(i))
            SubjectPerDatabase = numpy.array(SubjectPerDatabase)
            n = len(y_labels[i])
            Train_X.append(SubjectPerDatabase[i])
            num_pre = 0
            for num1 in range(i):
                num_pre += len(y_labels[num1])
            print("num_pre:", num_pre)
            slice_j1 = numpy.empty([0, 24, 224, 224, 3])
            for j in range(num_pre, num_pre + n):
                Train_X[0][j] = numpy.array(Train_X[0][j])
                # print("Train_X[0][j].shape:",Train_X[0][j].shape)
                # *slicing array*
                slice_j = Train_X[0][j][0:24, :, :, :]
                # print("slice:",slice_j.shape )
                slice_j = numpy.reshape(slice_j, [-1, 24, 224, 224, 3])
                # print("slice_j.shape:",slice_j.shape)
                slice_j1 = numpy.concatenate((slice_j, slice_j1), axis=0)
                # print("slice_j1.shape:",slice_j1.shape)
            Real_train_x = numpy.concatenate((Real_train_x, slice_j1), axis=0)
            Train_Y.append(y_labels[i])

        Train_Y_final = []
        for m in range(25):
            num = len(Train_Y[m])
            for n in range(num):
                # for n in range(24):
                # Train_X_final.append(Train_X[m][n])
                # Train_X_final=numpy.concatenate([Train_X_final,Train_X[m][n]],axis=0)
                Train_Y_final.append(Train_Y[m][n])
        Train_Y = numpy.array(Train_Y_final)
        Train_Y = np_utils.to_categorical(Train_Y, n_exp)

        print("----------------")
    else:
        for i in range(subjects):
            if subject == i:
                continue
            else:
                print("This is {}th subjects".format(i))
                SubjectPerDatabase = numpy.array(SubjectPerDatabase)
                n = len(y_labels[i])
                Train_X.append(SubjectPerDatabase[i])
                num_pre = 0
                for num1 in range(i):
                    num_pre += len(y_labels[num1])
                print("num_pre:", num_pre)
                slice_j1 = numpy.empty([0, 24, 224, 224, 3])
                for j in range(num_pre, num_pre + n):
                    Train_X[0][j] = numpy.array(Train_X[0][j])
                    # print("Train_X[0][j].shape:",Train_X[0][j].shape)
                    # *slicing array*
                    slice_j = Train_X[0][j][0:24, :, :, :]
                    # print("slice:",slice_j.shape )
                    slice_j = numpy.reshape(slice_j, [-1, 24, 224, 224, 3])
                    # print("slice_j.shape:",slice_j.shape)
                    slice_j1 = numpy.concatenate((slice_j, slice_j1), axis=0)
                    # print("slice_j1.shape:",slice_j1.shape)
                Real_train_x = numpy.concatenate((Real_train_x, slice_j1), axis=0)
                Train_Y.append(y_labels[i])

        Train_Y_final = []
        for m in range(25):
            num = len(Train_Y[m])
            for n in range(num):
                Train_Y_final.append(Train_Y[m][n])
        Train_Y = numpy.array(Train_Y_final)
        Train_Y = np_utils.to_categorical(Train_Y, n_exp)

        print("----------------")


    return Real_train_x, Train_Y, Real_test_x, Test_Y

for sub in range(subjects):

    print("\n\nStarting subject：" + str(sub) + "!")

    list_dir = os.listdir(inputDir)

    Train_X, Train_Y, Test_X, Test_Y = data_loader_with_LOSO(sub, SubperdB_list, labelperSub, subjects, n_exp)


    print("Train_X.shape:",Train_X.shape) #(246, 24, 224, 224, 3)
    print("Train_Y.shape:",Train_Y.shape)
    print("Test_X.shape:", Test_X.shape)
    print("Test_Y.shape:", Test_Y.shape)

    Train_X=numpy.transpose(Train_X, (0, 2, 3,4,1)) #(224,224,3,24)
    Test_X=numpy.transpose(Test_X, (0, 2, 3, 4, 1))

    # resnet3d
    from resnet3d import Resnet3DBuilder
    # model = Resnet3DBuilder.build_resnet_50((96, 96, 96, 1), 20)
    model = Resnet3DBuilder.build_resnet_18((224, 224, 3, 24), 7)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # model.fit(Train_X, Train_Y, batch_size=32)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    # Training the model
    hist = model.fit(Train_X, Train_Y, validation_data=(Test_X, Test_Y),
                     callbacks=callbacks_list, batch_size=1, epochs=5, shuffle=True)
    
    assert 0

    #vgg_16 3d
    model=VGG_16()


    # STEP2: 构建模型 MicroExpSTCNN Model  #unstable loss
    # model = Sequential()
    # model.add(Convolution3D(32, (3, 3, 3), input_shape=(24, 224, 224, 3),
    #                         data_format="channels_last", activation='relu'))
    # model.add(MaxPooling3D(pool_size=(3, 3, 3)))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(7, kernel_initializer='normal',name="dense_7"))
    # model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
    # model.summary()
    # model.load_weights('/home/michelexie/桌面/3D-micro-expression-recognition/dataset_and_weights/weights_microexpstcnn/weights-improvement-53-0.88.hdf5',by_name=True)
    # filepath = "/home/michelexie/桌面/3D-micro-expression-recognition/training_weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"


    filepath = "/home/michelexie/桌面/3D-micro-expression-recognition/training_weights/weights-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    # Training the model
    hist = model.fit(Train_X, Train_Y, validation_data=(Test_X, Test_Y),
                     callbacks=callbacks_list, batch_size=1, epochs=5, shuffle=True)

    # disp = plot_confusion_matrix(hist, Test_X, Test_Y, display_labels=None, cmap='viridis')
    # print(disp.confusion_matrix)


