"""
@author: Sharad Kumar Gupta, Vanshika Gupta
"""

import numpy as np
from keras.preprocessing.image import img_to_array, load_img, array_to_img
import glob
#from keras.models import *
from keras.models import Model
#from keras.layers import *
#from keras.optimizers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2
from skimage import io
import matplotlib.pyplot as plt
from keras.utils import plot_model
from sklearn.utils import class_weight
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from PIL import Image

np.random.seed(19680801)

class dataProcess(object):
    def __init__(self, out_rows, out_cols, data_path="./data/train/image", label_path="./data/train/label",
                 test_path="./data/test/image", testlbl_path="./data/test/label", npy_path="./npydata", log_dir="./tensorboard_net", img_type="tif"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.testlbl_path = testlbl_path
        self.npy_path = npy_path
        self.log_dir = log_dir

    def create_train_data(self):
        i=0
        print('Processing training images..')
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype = np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype = np.uint8)
        lbls = glob.glob(self.label_path + "/*." + self.img_type)
        for x in range(len(imgs)):
            imgpath = imgs[x]
            lblpath = lbls[x]
            #pic_name = imgpath.split('/')[-1]
            #labelpath = self.label_path + '/' + pic_name
            img = io.imread(imgpath, as_gray=False)
            label = io.imread(lblpath, as_gray=False)
            img = img.reshape((256,256,3))
            #img = load_img(imgpath, grayscale=False, target_size=[256, 256, 9])
            #label = load_img(lblpath, grayscale=False, target_size=[256, 256])
            #img = img_to_array(img)
            label = img_to_array(label)
            imgdatas[i] = img
            imglabels[i] = label
            if i%2 == 0:
                print ('Saved: {0}/{1} images'. format(i, len(imgs)))
            i += 1

        #print('Processing done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saved all training images to .npy files')

    def create_test_data(self):
        i=0
        j=0
        print('Processing test images..')
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)
        lbls = glob.glob(self.testlbl_path + "/*." + self.img_type)
        imglabels = np.ndarray((len(lbls), self.out_rows, self.out_cols, 1), dtype = np.uint8)
        testpathlist = []
        testlabelpathlist = []

        for imgname in imgs:
            testpath = imgname
            testpathlist.append(testpath)
            img = io.imread(testpath, as_gray=False)
            img = img.reshape((256,256,3))
            img = img_to_array(img)
            imgdatas[i] = img
            i += 1
            
        for x in range(len(lbls)):
            lblpath = lbls[x]
            testlabelpathlist.append(lblpath)
            label = io.imread(lblpath, as_gray=False, plugin='pil')
            label = label.reshape((256,256,1))
            #label = img_to_array(label)
            imglabels[j] = label
            j += 1

        txtname = './results/pic.txt'
        with open(txtname, 'w') as f:
            for i in range(len(testpathlist)):
                f.writelines(testpathlist[i] + '\n')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        
        txtlblname = './results/pic_label.txt'
        with open(txtlblname, 'w') as f:
            for j in range(len(testlabelpathlist)):
                f.writelines(testlabelpathlist[j] + '\n')
        np.save(self.npy_path + '/imgs_testlabel.npy', imglabels)
        print('Saved all test images and labels to .npy files')

    def load_train_data(self):
        print('Loading training images...')
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('uint8')
        imgs_mask_train = imgs_mask_train.astype('uint8')
        #imgs_train = imgs_train/255
        imgs_mask_train = imgs_mask_train/255
        imgs_mask_train[imgs_mask_train > 0] = 1
        imgs_mask_train[imgs_mask_train <= 0] = 0
        imgs_mask_train = imgs_mask_train.astype('uint8')
        print('Training images and labels loaded.')
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('Loading test images...')
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('uint8')
        lbl_test = np.load(self.npy_path + "/imgs_testlabel.npy")
        lbl_test = lbl_test.astype('uint8')
        
        print('Testing images and labels loaded.')
        #imgs_test = imgs_test/255
        return imgs_test, lbl_test


class loadNet(object):
    def __init__(self, img_rows=256, img_cols=256, training=False, log_dir="./tensorboard_net"):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.training = training
        self.log_dir = log_dir

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test, lbl_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test, lbl_test
        
    def load_net(self,n_classes=2, im_sz=256, n_channels=3, n_filters_start=32, growth_factor=2, upconv=True, class_weights=[0.05, 0.95]):
        droprate=0.25
        n_filters = n_filters_start
        inputs = Input((im_sz, im_sz, n_channels))
        #inputs = BatchNormalization()(inputs)
        conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #pool1 = Dropout(droprate)(pool1)
        
        n_filters *= growth_factor
        pool1 = BatchNormalization()(pool1)
        conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        pool2 = Dropout(droprate)(pool2)

        n_filters *= growth_factor
        pool2 = BatchNormalization()(pool2)
        conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        pool3 = Dropout(droprate)(pool3)

        n_filters *= growth_factor
        pool3 = BatchNormalization()(pool3)
        conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
        conv4_0 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_0)
        pool4_1 = MaxPooling2D(pool_size=(2, 2))(conv4_0)
        pool4_1 = Dropout(droprate)(pool4_1)
    
        n_filters *= growth_factor
        pool4_1 = BatchNormalization()(pool4_1)
        conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_1)
        conv4_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4_1)
        pool4_2 = MaxPooling2D(pool_size=(2, 2))(conv4_1)
        pool4_2 = Dropout(droprate)(pool4_2)

        n_filters *= growth_factor
        conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4_2)
        conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)
    
        n_filters //= growth_factor
        if upconv:
            up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4_1])
        else:
            up6_1 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4_1])
        up6_1 = BatchNormalization()(up6_1)
        conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_1)
        conv6_1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_1)
        conv6_1 = Dropout(droprate)(conv6_1)

        n_filters //= growth_factor
        if upconv:
            up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_1), conv4_0])
        else:
            up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_1), conv4_0])
        up6_2 = BatchNormalization()(up6_2)
        conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6_2)
        conv6_2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6_2)
        conv6_2 = Dropout(droprate)(conv6_2)
    
        n_filters //= growth_factor
        if upconv:
            up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6_2), conv3])
        else:
            up7 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3])
        up7 = BatchNormalization()(up7)
        conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = Dropout(droprate)(conv7)

        n_filters //= growth_factor
        if upconv:
            up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
        else:
            up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
        up8 = BatchNormalization()(up8)
        conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = Dropout(droprate)(conv8)
    
        n_filters //= growth_factor
        if upconv:
            up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
        else:
            up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)
    
        conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)
    
        model = Model(inputs=inputs, outputs=conv10)
    
        def weighted_binary_crossentropy(y_true, y_pred):
            class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])
            return K.sum(class_loglosses * K.constant(class_weights))

        model.compile(optimizer=Adam(), loss=weighted_binary_crossentropy, metrics=['accuracy'])
        if(self.training==False):
            model.load_weights("DeepMARSnet.hdf5")
        return model


    def train(self):
        print("Loading data...")
        imgs_train, imgs_mask_train, imgs_test, lbl_test = self.load_data()
        print("All data loaded.") 
        print("Searching working directory for DeepMARSnet...")
        model = self.load_net()
        print("DeepMARSnet found and loaded.")
        plot_model(model, to_file='CNN_Model_Architecture.png')
        model.summary()
        if(self.training==True):
            print('Training Required = True')
            model_checkpoint = ModelCheckpoint('DeepMARSnet.hdf5', monitor='loss', verbose=1, save_best_only=True)
            print('Training DeepMARSNet...')
            csv_logger = CSVLogger('log_net.csv', append=True, separator=';')
            tensorboard = TensorBoard(self.log_dir, write_graph=True, write_images=True)
            history = model.fit(imgs_train, imgs_mask_train, batch_size=16, epochs=20, verbose=1, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint, csv_logger, tensorboard])#, weighted_metrics=sample_weight)
            plt.figure(figsize=(20,10))
            plt.subplot(1, 2, 1)
            plt.suptitle('Optimizer : Adam', fontsize=10)
            plt.ylabel('Loss', fontsize=16)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.legend(loc='upper right')
            
            plt.subplot(1, 2, 2)
            plt.ylabel('Accuracy', fontsize=16)
            plt.plot(history.history['acc'], label='Training Accuracy')
            plt.plot(history.history['val_acc'], label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.show()
        else:
            print("Using pre-trained weights from network...")
        print("Predicting Test Results")
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/imgs_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("Converting test array to images...")
        imgs = np.load('./results/imgs_mask_test.npy')
        piclist = []
        for line in open("./results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        get_size = cv2.imread('./data/test/' + piclist[0]);
        height, width, channel = get_size.shape
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = imgs[i]
            #img = array_to_img(img[:,:,1])
            img = img[:,:,0]
            #img = img/255
            #img[img > 0.001] = 255
            #img[img <= 0.001] = 0
            #img = img.astype('uint8')
            img = Image.fromarray(img)
            img.save(path)
            #cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            #cv_pic = cv2.resize(cv_pic, (width, height), interpolation=cv2.INTER_CUBIC)
            #binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            #cv2.imwrite(path, cv_save)
        print("Conversion Complete")
        

if __name__ == '__main__':
    Data = dataProcess(256,256)
    Data.create_train_data()
    Data.create_test_data()
    net = loadNet()
    net_model = net.load_net()
    net.train()
    net.save_img()
    print("Process Completed Successfully")