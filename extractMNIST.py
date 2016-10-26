#!/bin/python
'''
Author: QIN Shuo
Date: 2016/9/27
Description:
    THis file illustrate how to extract mnist dataset
    http://yann.lecun.com/exdb/mnist/

    caffe use protobuf to read data, data structure is datum,
    defined in caffe.proto. Google protobuf will compile the 
    .proto file and assign a SerializeToString like function. 
    I inspect the caffe generated lmdb file, there is a strange
    string (str_id mybe) at the front of the data. It's better to use 
    caffe.proto.caffe_pb2.Datum()
'''

import os
import sys
import struct # interpret bytes as packed binary data

import lmdb
import numpy as np
from PIL import Image




class ExtractMinst():
    def __init__(self):
        self._dbname = 'mnist_train'
        self._dataname = "  "
        self._labelname = " " 
        self.pixels = ""
        self.labels = ""
        self.num_img = 0
        self.num_cols = 0
        self.num_rows = 0
    def SetSrcFileName(self,name):
        self._dataname = name
        print "Data file name is: ",self._dataname
    def SetLabelName(self,name):
        self._labelname = name
        print "Label file name is ", self._labelname

    def SetDBName(self,name):
        '''
        Set output lmdb file name 
        '''
        self._dbname = name
        print "Data file name is: ",self._dbname

    def Read(self):
        self.ReadData()
        self.ReadLabel()

    def ReadData(self):
        print "Reading..."
        # check file existance
        if(not os.path.isfile(self._dataname)):
            print self._dataname," does not exist.."
            return
        # read image data file
        data_file = open(self._dataname,"rb")
        content = data_file.read()

        # 1-2: magic
        magic = content[0:4]
        magic = struct.unpack('>i',magic)[0]

        # 3rd byte as type
        # it is useless here, pixel type is fixed to unsigned byte
        tp = content[3]
        if (tp == '\x08'):
            type = "unsigned byte"
            typeid='B'
        elif (tp=='\0x09'):
            type = "signed byte"
            typeid='b'
        elif (tp=='\0x0B'):
            type = "short (2 byte)"
            typeid='h'
        elif (tp=='\0x0C'):
            type = "int (4 bytes)"
            typeid='i'
        elif (tp=='\0x0D'):
            type = "float (4 bytes)"
            typeid='f'
        elif (tp=='\0x0E'):
            type = "double (8 byte)"
            typeid='d'
        else:
            type="unknown"
            typeid='P'
        print "data type is: ",type
        
        #4 rd for number of images
        self.num_img = struct.unpack('>i',content[4:8])[0] # convert from tuple to number
        print "number of images is: ", self.num_img 
    
        # number of rows
        self.num_rows = struct.unpack('>i',content[8:12])[0] # convert from tuple to number
        print "number of rows is: ", self.num_rows
        
        # number of colums
        self.num_cols = struct.unpack('>i',content[12:16])[0] # convert from tuple to number
        print "number of cols is: ", self.num_cols

        # extract pixels
        print "converting UnsighedByte to float"
        pixel_temp = content[16:]
        self.pixels = pixel_temp
        #for px in pixel_temp:
        #    tt = str(float(struct.unpack('B',px)[0]))
        #    self.pixels = self.pixels+tt

    def ReadLabel(self):
        print "Reading labels..."
        # check file existance
        if (not os.path.isfile(self._labelname)):
            print self._labelname," does not exist.."
            return
        
        # read label data file
        label_file = open(self._labelname,"rb")
        content = label_file.read()

        # 1-2: magic
        magic = content[0:4]
        magic = struct.unpack('>i',magic)[0]
        
        #4 rd for number of items
        num_lab = struct.unpack('>i',content[4:8])[0] # convert from tuple to number
        print "number of labels is: ", num_lab 

        # extract label
        print "converting UnsighedByte to int"
        label_temp = content[8:]
        self.labels= label_temp
        #for lab in label_temp:
        #    tt = str(int(struct.unpack('B',lab)[0]))            
        #    self.labels = self.labels+tt
        

    def Write_to_lmdb(self):
        '''
        write data to a lmdb file. All images and labels are stacked together
        entries:
        1. channels
        2. height
        3. width
        4. num: total count of numbers
        5. data: image data
        6. label: corresponding label
        '''
        env = lmdb.open(self._dbname,map_size=1024*1024*1024)
        size = cnv.num_img
        with env.begin(write=True) as txn: # Transaction commits automatically:
            txn.put('channel', '1')
            txn.put('height',str(self.num_cols))
            txn.put('width', str(self.num_rows))
            txn.put('num',   str(self.num_img))
            txn.put('data',  self.pixels)
            txn.put('label', self.labels)    
            #txn.commit()
            print 'All entries are: ',txn.stat()['entries']

        print 'LMDB environment information: ',env.info()
        print "Max key size: ", str(env.max_key_size())
        env.close()
        print "Writing Done!"


def print_label(cnv):
    file = open("my_read_label.txt",'w')
    for i,lab in enumerate(cnv.label):
        file.write(str(i) + " : " +  hex(ord(lab))+"\n")
    file.close()

def print_first(cnv):
    file = open("my_read_image.txt",'w')
    vv = cnv.pixels[0:cnv.num_cols*cnv.num_rows]
    out = "\n".join("{:02x}".format(ord(c)) for c in vv)
    file.write(out)
    file.close()

# save image by index    
def save_image(cnv,index,name='testfile.PNG'):
    if index >= cnv.num_img:
        print "Index out of range.."
        return
    # extract
    imglen = cnv.num_cols*cnv.num_rows
    img = cnv.pixels[imglen*index:imglen*(index+1)]
    img2 = np.zeros(cnv.num_rows*cnv.num_cols)
    for id, px in enumerate(img):
        img2[id] = ord(px)
    
    img2 = img2.reshape((cnv.num_cols,cnv.num_rows))

    im = Image.fromarray(img2)
    im = im.convert('RGB')
    im.save(name)


if __name__=='__main__':
    cnv = ExtractMinst()
    cnv.SetSrcFileName("data/test-images.idx3-ubyte")
    cnv.SetLabelName("data/test-labels.idx1-ubyte")
    cnv.SetDBName("mnist_test")
    cnv.Read()

    cnv.Write_to_lmdb()
    
