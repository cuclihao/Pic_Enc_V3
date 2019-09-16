#!/usr/bin/env python
# coding: utf-8

# In[37]:


##====================================================================================
#All copyrights are reserved by Hao Li. E-mial:cuclihao@cuc.edu.cn
#All following source code is free to distribute, to use, and to modify
#    for research and study purposes, but absolutely NOT for commercial uses.
#If you use any of the following code in your academic publication(s), 
#    please cite the corresponding paper. 
#If you have any questions, please email me and I will try to response you ASAP.
##====================================================================================
import time
import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PIL import Image
import hashlib
get_ipython().run_line_magic('matplotlib', 'inline')

#confuse and diffuse scheme
#date : 2019.6.3 10.56

def PL_PWLCM(y0,p2,exp32):
    y1=0
    if y0<p2:
        y1 = math.floor(exp32*y0/p2)
    elif p2<y0 and y0 < exp32/2:
        y1 = math.floor(exp32*(y0-p2)/(exp32/2 - p2))
    elif y0 == exp32/2 :
        y1 = 0
    else:
        y1 = PL_PWLCM(exp32-y0, p2, exp32)
    return y1

def PL_Logistic(x0,exp32):
    x1 = math.floor(4*x0*(exp32-x0)/exp32)
    return x1

def PL_PWLCM_Logistic(x0,y0,p1,z0,p2):
    exp32 = 4294967296
    y1 = PL_PWLCM(y0,p1,exp32)
    z1 = PL_PWLCM(z0,p2,exp32)
    x1 = PL_Logistic(x0,exp32)
    temp1 = x1 ^ y1
    r1 = (temp1 + z1) % exp32
    return x1,y1,z1,r1

def PWLCM_Init(x0,y0,p1,p2,exp32,n):
    x1 = x0
    y1 = y0
    for i in range(n):
        x1 = PL_PWLCM(x1,p1,exp32)
        y1 = PL_PWLCM(y1,p2,exp32)
    return x1,y1

def Logistic_Init(x0,n,exp32):
    x1 = x0
    for i in range(n):
        x1 = PL_Logistic(x1,exp32)
    return x1

def InitPRNG(x0,y0,p1,z0,p2,n):
    exp32 = 4294967296
    for i in range(n):
        y1,z1 = PWLCM_Init(y0,z0,p1,p2,exp32,n)
        x1 = Logistic_Init(x0,n,exp32)
    return x1,y1,z1

# get the second column
def takeSecond(elem):
    return elem[1]
#key generation v2 output : PRNGlist, 32-Bit sorted sorting index 
def getPRNG(a,b,c,x0,y0,p1,z0,p2):
    iLen = math.ceil(a*b*c/4)
    SortList = list();
    ValueList = np.zeros(a*b*c, dtype = np.int, order = 'C')
    time_start=time.time()
    for iNum in range(iLen):
        x0,y0,z0,res = PL_PWLCM_Logistic(x0,y0,p1,z0,p2)
        SortList.append([iNum,res])
        ValueList[iNum*4] = (res >> 24) & 0xff
        ValueList[iNum*4+1] = (res >> 16) & 0x00ff
        ValueList[iNum*4+2] = (res >> 8) & 0x0000ff
        ValueList[iNum*4+3] = (res >> 0) & 0x000000ff
    time_end=time.time()
    #print('PRNG Generation time cost:',time_end-time_start,'s')
    SortList.sort(key=takeSecond)
    if c == 1: #gray
        return SortList,ValueList.reshape(a,b),x0,y0,z0
    else:
        return SortList,ValueList.reshape(a,b,c),x0,y0,z0

# Initialization
def getParas(key):
    p2 = key & 0x7fffffff
    z0 = (key>>31) & 0x7fffffff
    p1 = (key>>62) & 0x7fffffff
    y0 = (key>>93) & 0x7fffffff
    x0 = (key>>124) & 0xffffffff
    
    x0,y0,z0 = InitPRNG(x0,y0,p1,z0,p2,20)    
    return x0,y0,p1,z0,p2

def Enc(a,SortKey,ValueKey,scale,diffRound=1):
    w,h = a.shape
    k = SortKey
    #confusion
    c =  np.zeros((w,h), dtype=np.int)
    istep = h//2
    for i in range(len(k)):
            iRow = (k[i]//istep)*2
            iCol = (k[i] % istep)*2
            iRow0 = (i // istep)*2
            iCol0 = (i % istep)*2
            c[iRow0,iCol0] = a[iRow,iCol]
            c[iRow0+1,iCol0] = a[iRow+1,iCol]
            c[iRow0,iCol0+1] = a[iRow,iCol+1]
            c[iRow0+1,iCol0+1] = a[iRow+1,iCol+1]

    #扩散 diffuse
    b = np.zeros((w,h), dtype=np.int)
    b = c
    for iwhole in range(diffRound):
        #step2.1 diffuse row
        for iRow in range(0,w,1):
            if iRow == 0:
                b[iRow,:] = (b[-1,:] + b[iRow,:] + ValueKey[iRow,:]) % scale
            else:
                b[iRow,:] = (b[iRow-1,:] + b[iRow,:] + ValueKey[iRow,:]) % scale

        #step2.2 diffuse column
        for iCol in range(0,h,1):
            if iCol == 0:
                b[:, iCol] = (b[:, -1] + b[:, iCol] + ValueKey[:, iCol]) % scale
            else:
                b[:, iCol] = (b[:, iCol-1] + b[:, iCol] + ValueKey[:, iCol]) % scale

    return c


def Dec(a,SortKey,ValueKey,scale,diffRound=1):
    #step1: de-confusion
    w,h = a.shape
    c =  np.zeros((w,h), dtype=np.int)
    c = a
    k = SortKey
    for iwhole in range(diffRound):
        for iCol in range(h-1,-1,-1):
            if iCol == 0:
                c[:, iCol] = (c[:, iCol] - c[:, -1] - ValueKey[:, iCol]) % scale
            else:
                c[:, iCol] = (c[:, iCol] - c[:, iCol-1] - ValueKey[:, iCol]) % scale
        for iRow in range(w-1,-1,-1):
            if iRow == 0:
                c[iRow,:] = (c[iRow,:] - c[-1,:] - ValueKey[iRow,:])% scale
            else:
                c[iRow,:] = (c[iRow,:] - c[iRow-1,:] - ValueKey[iRow,:])% scale 
    
    b =  np.zeros((w,h), dtype=np.int)
    istep = h//2
    #print("step:",istep)
    for i in range(len(k)):
            iRow = (k[i]//istep)*2
            iCol = (k[i] % istep)*2
            iRow0 = (i // istep)*2
            iCol0 = (i % istep)*2
            b[iRow,iCol] = c[iRow0,iCol0]
            b[iRow+1,iCol] = c[iRow0+1,iCol0]
            b[iRow,iCol+1] = c[iRow0,iCol0+1]
            b[iRow+1,iCol+1] = c[iRow0+1,iCol0+1]
    
    return b


def DoEnc(img,k):
    n = 2 # Encryption round
    w,h = img.shape #return row and column
    for i in range(n):
        x0,y0,p1,z0,p2 = getParas(k[i])
        sort,valuekey,x0,y0,z0 = getPRNG(w,h,1,x0,y0,p1,z0,p2)
        arr = np.array(sort)
        sortkey = arr[:,0]
        img = Enc(img,sortkey,valuekey,256,2)
    return img


def DoDec(img,k):
    n = 2 # Encryption round    
    w,h = img.shape #return row and column
    for i in range(n-1,-1,-1):
        x0,y0,p1,z0,p2 = getParas(k[i])
        sort,valuekey,x0,y0,z0 = getPRNG(w,h,1,x0,y0,p1,z0,p2)
        arr = np.array(sort)
        sortkey = arr[:,0]
        img = Dec(img,sortkey,valuekey,256,2)
    return img

def SpeedTest():
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B]
    
    icount = 0
    index = 0
    path = './test/'
    pictures = os.listdir(path)
    for picName in pictures:
        index += 1

    plt.figure(figsize=(26,9*index),dpi=80)
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    for picName in pictures:
        picPath = path + picName
        lenna_img = np.array(Image.open(picPath).convert('L'))
        print(picName,"'s size is :", lenna_img.shape)
    
        plt.subplot(index,3,1+icount*3),plt.imshow(lenna_img,cmap="gray")
        plt.title("Origin image")
        
        time_start=time.time()
        lenna_cipher = DoEnc(lenna_img,key)
        time_end=time.time()
    
        plt.subplot(index,3,2+icount*3),plt.imshow(lenna_cipher,cmap="gray")
        plt.title("Enc image")
        
        time_DecStart=time.time()
        lenna_dec = DoDec(lenna_cipher,key)
        time_DecEnd=time.time()

        plt.subplot(index,3,3+icount*3),plt.imshow(lenna_dec,cmap="gray")
        plt.title("Dec image")
    
        print("Function Enc cost time:", time_end-time_start)
        print("Function Dec cost time:", time_DecEnd-time_DecStart)
        icount += 1    
    return 0

SpeedTest()


# In[38]:


#Visual Perception
import os
path = "./test/"
pictures = os.listdir(path)
key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
icount = 0
index = 0
for picName in pictures:
    index += 1

plt.figure(figsize=(26,9*index),dpi=80)
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

for picName in pictures:
    picPath = path + picName
    lenna_img = np.array(Image.open(picPath).convert('L'))
    plt.subplot(index,3,1+icount*3),plt.imshow(lenna_img,cmap="gray")
    plt.title(picName)
    
    lenna_cipher = DoEnc(lenna_img,key)
    plt.subplot(index,3,2+icount*3),plt.imshow(lenna_cipher,cmap="gray")
    plt.title("Encrypted "+picName)
    
    lenna_dec = DoDec(lenna_cipher,key)  
    plt.subplot(index,3,3+icount*3),plt.imshow(lenna_dec,cmap="gray") 
    plt.title("Decrypted "+picName)
    icount += 1


# In[39]:


#Histograms Analysis
def expectation(x,n):
    tatal = 0.0
    for i in range(n):
        tatal = tatal + x[i]
    E = tatal/n
    return E

def Variance(x,n):
    varValue = 0.0
    expValue = expectation(x,n)
    for i in range(n):
        varValue = varValue + math.pow(x[i]-expValue,2)
    return varValue/n

icount = 0
index = 0
path = './test/'
pictures = os.listdir(path)
for picName in pictures:
    index += 1

plt.figure(figsize=(26,9*index),dpi=80)
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 

for picName in pictures:
    picPath = path + picName
    lenna_img = np.array(Image.open(picPath).convert('L'))
    arr=lenna_img.flatten()
    varValue = Variance(arr,arr.size)
    plt.subplot(index,2,1+icount*2)
    n, bins, patches = plt.hist(arr, bins=256, density=0, facecolor='gray', alpha=0.75)  
    plt.title(picName)
    plt.xlabel('gray Value')
    plt.ylabel('Distribution')
    
    lenna_cipher = DoEnc(lenna_img,key)    
    arr=lenna_cipher.flatten()
    varValue = Variance(arr,arr.size)
    plt.subplot(index,2,2+icount*2)
    n, bins, patches = plt.hist(arr, bins=256, density=0, facecolor='gray', alpha=0.75)
    plt.title("Encrypted "+picName)
    plt.xlabel('gray Value')
    plt.ylabel('Distribution')
    
    print("encrypted", picName, "'s variance is ",str(varValue))
    
    icount += 1


# In[40]:


# Chi-square analysis
def getFre(value,img):
    a,b = img.shape
    icount = 0.0
    for i in range(a):
        for j in range(b):
            if img[i][j] == value:
                icount += 1
    return icount
def ChiSquare(img):
    a,b = img.shape
    print("image's shape", img.shape)
    e = a*b/256
    res = 0
    for i in range(256):
        oi = getFre(i,img)
        temp = math.pow(oi-e,2) / e
        res += temp
    print("Chi-square value:", res)
    return 0

def calcChiSquareList():
    path = "./test/"
    pictures = os.listdir(path)
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4032E9F680BC614B] 
    icount = 0
    for picName in pictures:
        picPath = path + picName
        print(picName)
        lenna_img = np.array(Image.open(picPath).convert('L'))
        lenna_cipher = DoEnc(lenna_img,key)
        ChiSquare(lenna_cipher)
    return 0
        
calcChiSquareList()


# In[41]:


#Correlation Analysis
from PIL import Image
from matplotlib.widgets import Cursor
import numpy as np
import matplotlib.pyplot as plt
import math

def expectation(x,n):
    #print(x)
    tatal = 0.0
    for i in range(n):
        tatal = tatal + x[i]
    E = tatal/n
    #print(E)
    return E

def Variance(x,n):
    varValue = 0.0
    expValue = expectation(x,n)
    for i in range(n):
        varValue = varValue + math.pow(x[i]-expValue,2)
    return varValue/n

def cov(x,y,n):
    covValue = 0.0
    expX = expectation(x,n)
    expY = expectation(y,n)
    for i in range(n):
        covValue = covValue + (x[i] - expX)*(y[i]-expY)
    return covValue/n
def corr_coefficient(x,y,n):
    covValue = cov(x,y,n)
    varX = Variance(x,n)
    varY = Variance(y,n)
    varX = math.sqrt( varX )
    varY = math.sqrt( varY )
    #print(covValue,varX,varY)
    return  covValue/(varX*varX)

def Correlation(img,img2,LineColor,itype):
    x0=[]
    x1=[]
    y0=[]
    y1=[]
    if itype == 'Horizontal':
        for a in range(128):
            for b in range(32):
                x0.append(img2[2*a][4*b])
                x1.append(img2[1+2*a][4*b])
                y0.append(img[2*a][4*b])
                y1.append(img[1+2*a][4*b])
    elif itype == 'Vertical':
        for a in range(32):
            for b in range(128):
                x0.append(img2[4*a][2*b])
                x1.append(img2[4*a][1+2*b])
                y0.append(img[4*a][2*b])
                y1.append(img[4*a][1+2*b])
    else:
        for a in range(128):
            for b in range(32):
                x0.append(img2[2*a][4*b])
                x1.append(img2[1+2*a][1+4*b])
                y0.append(img[2*a][4*b])
                y1.append(img[1+2*a][1+4*b])
    return x0,x1,y0,y1

def correlation(plain,cipher):
    r=plain
    r1=cipher
    x0,x1,y0,y1 = Correlation(r,r1,'gray','Horizontal')
    fig = plt.figure(figsize=(34,10),dpi=100)
    ax = fig.add_subplot(1,2,1, facecolor='#EBEBEB')
    ax.plot(y0, y1, '.',color='gray')
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    plt.title('Horizontal')
    plt.xlabel("x(m,n)")
    plt.ylabel("x(m,n+1)")
    # Set useblit=True on most backends for enhanced performance.
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    ax = fig.add_subplot(1,2,2, facecolor='#EBEBEB')
    ax.plot(x0, x1, '.',color='gray')
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    plt.title('Encrypted-Horizontal')
    plt.xlabel("x(m,n)")
    plt.ylabel("x(m,n+1)")
    # Set useblit=True on most backends for enhanced performance.
    cursor = Cursor(ax, useblit=True, color='gray', linewidth=2)
    return 0


def drawFig():
    icount = 0
    index = 0
    path = './test/'
    pictures = os.listdir(path)
    for picName in pictures:
        index += 1

    fig = plt.figure(figsize=(30,17*index),dpi=80)
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 

    for picName in pictures:
        picPath = path + picName
        lenna_img = np.array(Image.open(picPath).convert('L'))
        lenna_cipher = DoEnc(lenna_img,key)
        lenna_img = np.array(Image.open(picPath).convert('L'))
        plt.subplot(index*2,4,1+icount*8),plt.imshow(lenna_img,cmap="gray")
        plt.title(picName)
        plt.subplot(index*2,4,5+icount*8),plt.imshow(lenna_cipher,cmap="gray")
        plt.title("encrypted "+picName)
        
        x0,x1,y0,y1 = Correlation(lenna_img,lenna_cipher,'gray','Horizontal')
        ax = fig.add_subplot(index*2,4,2+icount*8, facecolor='#EBEBEB')
        ax.plot(y0, y1, '.',color='gray')
        ax.set_xlim(-10, 270)
        ax.set_ylim(-10, 270)
        plt.title('Horizontal')
        plt.xlabel("x(m,n)")
        plt.ylabel("x(m,n+1)")
        cursor = Cursor(ax, useblit=True, color='red', linewidth=2)        
        ax = fig.add_subplot(index*2,4,6+icount*8, facecolor='#EBEBEB')
        ax.plot(x0, x1, '.',color='gray')
        ax.set_xlim(-10, 270)
        ax.set_ylim(-10, 270)
        plt.title('Encrypted-Horizontal')
        plt.xlabel("x(m,n)")
        plt.ylabel("x(m,n+1)")
        # Set useblit=True on most backends for enhanced performance.
        cursor = Cursor(ax, useblit=True, color='gray', linewidth=2)
        
        x0,x1,y0,y1 = Correlation(lenna_img,lenna_cipher,'gray','Vertical')
        ax = fig.add_subplot(index*2,4,3+icount*8, facecolor='#EBEBEB')
        ax.plot(y0, y1, '.',color='gray')
        ax.set_xlim(-10, 270)
        ax.set_ylim(-10, 270)
        plt.title('Vertical')
        plt.xlabel("x(m,n)")
        plt.ylabel("x(m+1,n)")
        cursor = Cursor(ax, useblit=True, color='red', linewidth=2)        
        ax = fig.add_subplot(index*2,4,7+icount*8, facecolor='#EBEBEB')
        ax.plot(x0, x1, '.',color='gray')
        ax.set_xlim(-10, 270)
        ax.set_ylim(-10, 270)
        plt.title('Encrypted-Vertical')
        plt.xlabel("x(m,n)")
        plt.ylabel("x(m,n+1)")
        # Set useblit=True on most backends for enhanced performance.
        cursor = Cursor(ax, useblit=True, color='gray', linewidth=2)
        
        x0,x1,y0,y1 = Correlation(lenna_img,lenna_cipher,'gray','Diagonal')
        ax = fig.add_subplot(index*2,4,4+icount*8, facecolor='#EBEBEB')
        ax.plot(y0, y1, '.',color='gray')
        ax.set_xlim(-10, 270)
        ax.set_ylim(-10, 270)
        plt.title('Diagonal')
        plt.xlabel("x(m,n)")
        plt.ylabel("x(m+1,n+1)")
        cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
        
        ax = fig.add_subplot(index*2,4,8+icount*8, facecolor='#EBEBEB')
        ax.plot(x0, x1, '.',color='gray')
        ax.set_xlim(-10, 270)
        ax.set_ylim(-10, 270)
        plt.title('Encrypted-Diagonal')
        plt.xlabel("x(m,n)")
        plt.ylabel("x(m,n+1)")
        # Set useblit=True on most backends for enhanced performance.
        cursor = Cursor(ax, useblit=True, color='gray', linewidth=2)
        
        icount += 1
    return 0
 
def calc(img,itype):
    arrarR = img
    x,y = arrarR.shape
    # choose 5000 random position (row and col)
    np.random.seed(1111)
    position = np.random.randint(255,size=(5000,2))
    #print(position)
    
    if itype == 'Horizontal':
        value1 = arrarR[position[0:,0],position[0:,1]] 
        value2 = arrarR[position[0:,0],position[0:,1]+1] 
    elif itype == 'Vertical':
        value1 = arrarR[position[0:,0],position[0:,1]] 
        value2 = arrarR[position[0:,0]+1,position[0:,1]] 
    else:
        value1 = arrarR[position[0:,0],position[0:,1]] 
        value2 = arrarR[position[0:,0]+1,position[0:,1]+1] 
    return corr_coefficient(value1,value2,5000)

def calcList():
    path = "./test/"
    pictures = os.listdir(path)
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B]
    icount = 0
    for picName in pictures:
        picPath = path + picName
        print(picName,"'s correlation coefficient:")
        lenna_img = np.array(Image.open(picPath).convert('L'))
        print("Horizontal             Vertical            Diagonal")
        x1=calc(lenna_img,'Horizontal')
        x2=calc(lenna_img,'Vertical')
        x3=calc(lenna_img,'Diagonal')
        print(x1,x2,x3)
        print("Encrypted",picName,"'s correlation coefficient:")
        lenna_cipher = DoEnc(lenna_img,key)
        print("Horizontal             Vertical            Diagonal")
        x1=calc(lenna_cipher,'Horizontal')
        x2=calc(lenna_cipher,'Vertical')
        x3=calc(lenna_cipher,'Diagonal')
        print(x1,x2,x3)
 
calcList()
drawFig()


# In[42]:


#information entropy
def entropy(img):
    tmp = np.zeros(256)
    val = 0
    k = 0
    res = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res

def infEntropy(plain,cipher):
    print("Information Entropy:")
    r0 = plain
    res0 = entropy(r0)
    r1 = cipher
    res1 = entropy(r1)
    print("Pic Entropy is:"+ str(res0)+ "\nThen,Encrypted Pic Entropy is:"+str(res1))
    return 0

path = "./test/"
pictures = os.listdir(path)
key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
icount = 0
for picName in pictures:
    picPath = path + picName
    print(picName,"'s information entropy:")
    lenna_img = np.array(Image.open(picPath).convert('L'))
    lenna_cipher = DoEnc(lenna_img,key)
    lenna_img = np.array(Image.open(picPath).convert('L'))
    infEntropy(lenna_img,lenna_cipher)


# In[43]:


#Image Sensitivity
def NPCR(n1,n2):
    m,n = n1.shape
    D = 0.0
    T=m*n
    #print(m,n,T)
    iCount = 0
    for iRow in range(m):
        for iCol in range(n): #column
            if n1[iRow][iCol] != n2[iRow][iCol]:
                D = float(D+1)
                iCount = iCount+1
    #print(iCount)
    return D/T

def UACI(n1,n2,F):
    m,n = n1.shape
    T = m*n
    #denominator
    Den = F*T
    Dif = 0.0
    iCount = 0
    for iRow in range(m):
        for iCol in range(n): #column
            Dif = Dif + (abs(int(n1[iRow][iCol]) - int(n2[iRow][iCol])))
            iCount = iCount+1
    return Dif/Den

def GetSlightImg(img):
    w,h = img.shape
    randRow = np.random.randint(h)
    randCol = np.random.randint(w)
    img[randRow][randCol] = img[randRow][randCol] ^ 0x1    
    return img

def testNPCRandUACI():
    path = "./test/"  
    pictures = os.listdir(path)
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
    print("Image Sensitivity:")
    for picName in pictures:
        picPath = path + picName
        print(picName,":")
        iCount = 0
        NPCRvalue = 0.0
        UACIvalue = 0.0
        for i in range(0,5,1):# test 50 times using different keys 
            iCount += 1
            #print("the ",iCount,"th time (50 times in total)")
            key = key + [0,1]
            lenna_img = np.array(Image.open(picPath).convert('L'))
            lenna_cipher = DoEnc(lenna_img,key)

            lenna_img = np.array(Image.open(picPath).convert('L'))
            lenna_img2 = GetSlightImg(lenna_img)
            lenna_cipher2 = DoEnc(lenna_img2,key)
            
            vtemp1 = NPCR(lenna_cipher,lenna_cipher2)
            NPCRvalue = NPCRvalue + vtemp1
            vTemp2 = UACI(lenna_cipher,lenna_cipher2,255)
            UACIvalue = UACIvalue + vTemp2
        
        NPCRvalue = NPCRvalue / 5
        UACIvalue = UACIvalue / 5
        print("average NPCR (5 times):",NPCRvalue)
        print("average UACI (5 times):",UACIvalue)
    return 0 

testNPCRandUACI()


# In[44]:


#key sensitivity and key space
def difffuc(n1,n2):
    m,n = n1.shape
    D = 0.0
    T=m*n
    iCount = 0
    for iRow in range(m):
        for iCol in range(n): #column
            if n1[iRow][iCol]!= n2[iRow][iCol]:
                D = float(D+1)
            iCount = iCount+1
    return D/T
def KeySensitivity():
    key1 = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
    key2 = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC615B] 
    
    iCount = 0
    index = 0
    path = './test/'
    pictures = os.listdir(path)
    for picName in pictures:
        index += 1

    plt.figure(figsize=(25,15*index),dpi=80)
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18    
    print("Key Sensitivity:")

    for picName in pictures:
        picPath = path + picName
        lenna_img = np.array(Image.open(picPath).convert('L'))
        plt.subplot(index*2,3,1+iCount*6),plt.imshow(lenna_img,cmap="gray")
        plt.title("a: "+picName)
        
        lenna_cipher1 = DoEnc(lenna_img,key1)
        plt.subplot(index*2,3,2+iCount*6),plt.imshow(lenna_cipher1,cmap="gray")
        plt.title("b: Encrypted by key1")

        lenna_cipher2 = DoEnc(lenna_img,key2)
        plt.subplot(index*2,3,3+iCount*6),plt.imshow(lenna_cipher2,cmap="gray")
        plt.title("c: Encrypted by key2")

        lenna_dec1 = DoDec(lenna_cipher1,key1)
        plt.subplot(index*2,3,4+iCount*6),plt.imshow(lenna_dec1,cmap="gray")
        plt.title("d: Dec image - (Enc:key1,Dec:key1)")

        lenna_dec2 = DoDec(lenna_cipher1,key2)
        plt.subplot(index*2,3,5+iCount*6),plt.imshow(lenna_dec2,cmap="gray")
        plt.title("e: Dec image - (Enc:key1,Dec:key2)")

        diff = difffuc(lenna_cipher1,lenna_cipher2)
        plt.subplot(index*2,3,6+iCount*6),plt.imshow(lenna_cipher1-lenna_cipher2,cmap="gray")
        plt.title("f: Difference = "+str(diff*100)+"%")
        iCount += 1
    return 0

KeySensitivity()


# In[46]:


#CPA
def NPCRandUACI():
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
    print("Image Sensitivity:")
    iCount = 0
    NPCRvalue = 0.0
    UACIvalue = 0.0
    print("all white pic:")
    for i in range(0,5,1):# test 5 times using different keys 
        iCount += 1
        #print("the ",iCount,"th time (5 times in total)")
        key = key + [0,1]
        
        lenna_img = np.zeros((256,256),dtype=np.int)
        lenna_img[::]=255
        

        lenna_cipher = DoEnc(lenna_img,key)
        
        lenna_img = np.zeros((256,256),dtype=np.int)
        lenna_img[::]=255
        lenna_img2 = GetSlightImg(lenna_img)
        
        lenna_cipher2 = DoEnc(lenna_img2,key)
        
        vtemp1 = NPCR(lenna_cipher,lenna_cipher2)
        NPCRvalue = NPCRvalue + vtemp1
        vTemp2 = UACI(lenna_cipher,lenna_cipher2,255)
        UACIvalue = UACIvalue + vTemp2

    NPCRvalue = NPCRvalue / 5
    UACIvalue = UACIvalue / 5
    print("average NPCR (5 times):",NPCRvalue)
    print("average UACI (5 times):",UACIvalue)
    
    print("all black pic:")
    NPCRvalue = 0.0
    UACIvalue = 0.0
    for i in range(0,5,1):# test 5 times using different keys 
        iCount += 1
        #print("the ",iCount,"th time (5 times in total)")
        key = key + [0,1]
        
        lenna_img = np.zeros((256,256),dtype=np.int)
        
        lenna_cipher = DoEnc(lenna_img,key)
        
        lenna_img = np.zeros((256,256),dtype=np.int)
        lenna_img2 = GetSlightImg(lenna_img)
        
        lenna_cipher2 = DoEnc(lenna_img2,key)
        
        vtemp1 = NPCR(lenna_cipher,lenna_cipher2)
        NPCRvalue = NPCRvalue + vtemp1
        vTemp2 = UACI(lenna_cipher,lenna_cipher2,255)
        UACIvalue = UACIvalue + vTemp2

    NPCRvalue = NPCRvalue / 5
    UACIvalue = UACIvalue / 5
    print("average NPCR (5 times):",NPCRvalue)
    print("average UACI (5 times):",UACIvalue)
    return 0

def calcCoff():
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
    print("Correlation Analysis")
    print("all black pig:")
    lenna_img = np.zeros((256,256),dtype=np.int)
    print("Horizontal             Vertical            Diagonal")
    x1=calc(lenna_img,'Horizontal')
    x2=calc(lenna_img,'Vertical')
    x3=calc(lenna_img,'Diagonal')
    print(x1,x2,x3)
    print("Encrypted all black pic's correlation coefficient:")
    
    lenna_cipher = DoEnc(lenna_img,key)
    print("Horizontal             Vertical            Diagonal")
    x1=calc(lenna_cipher,'Horizontal')
    x2=calc(lenna_cipher,'Vertical')
    x3=calc(lenna_cipher,'Diagonal')
    print(x1,x2,x3)
    
    print("all white pig:")
    lenna_img = np.zeros((256,256),dtype=np.int)
    lenna_img[::] = 255
    print("Horizontal             Vertical            Diagonal")
    x1=calc(lenna_img,'Horizontal')
    x2=calc(lenna_img,'Vertical')
    x3=calc(lenna_img,'Diagonal')
    print(x1,x2,x3)
    print("Encrypted all white pic's correlation coefficient:")
    
    lenna_cipher = DoEnc(lenna_img,key)
    print("Horizontal             Vertical            Diagonal")
    x1=calc(lenna_cipher,'Horizontal')
    x2=calc(lenna_cipher,'Vertical')
    x3=calc(lenna_cipher,'Diagonal')
    print(x1,x2,x3)
    
    return 0

def testCPA():
    #all 0 pic
    plt.figure(figsize=(35,26),dpi=80)
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B] 
    
    # all black
    blackImg = np.zeros((256,256),dtype=np.int)
    print("the image's size is :", blackImg.shape)    
    plt.subplot(2,3,1),plt.imshow(blackImg,cmap="gray")
    plt.title("all black")
    
    blackImg_cipher = DoEnc(blackImg,key)    
    plt.subplot(2,3,2),plt.imshow(blackImg_cipher,cmap="gray")
    plt.title("Enc image")
    
    arr=blackImg_cipher.flatten() 
    varValue = Variance(arr,arr.size)    
    print("encrypted all black image's variance is ",str(varValue))
    plt.subplot(2,3,3)
    n, bins, patches = plt.hist(arr, bins=256, density=0, facecolor='gray', alpha=0.75)  
    plt.title("Encrypted all black image")
    plt.xlabel('Value')
    plt.ylabel('Distribution')
    
    # all white
    whiteImg = np.zeros((256,256),dtype=np.int)
    whiteImg[::]=255
    
    whiteImg2 = np.zeros((256,256,3),dtype=np.int)
    whiteImg2[::]=255
    plt.subplot(2,3,4),plt.imshow(whiteImg2,cmap="gray")
    plt.title("all white")
    
    whiteImg_cipher = DoEnc(whiteImg,key) 
    plt.subplot(2,3,5),plt.imshow(whiteImg_cipher,cmap="gray")
    plt.title("Enc image")
    
    arr=whiteImg_cipher.flatten() 
    varValue = Variance(arr,arr.size)    
    print("encrypted all white image's variance is ",str(varValue))
    
    plt.subplot(2,3,6)
    n, bins, patches = plt.hist(arr, bins=256, density=0, facecolor='gray', alpha=0.75)  
    plt.title("Encrypted all white image")
    plt.xlabel('Value')
    plt.ylabel('Distribution')

    #Correlation Analysis
    calcCoff()
    
    infEntropy(blackImg,blackImg_cipher)
    infEntropy(whiteImg,whiteImg_cipher)
    
    NPCRandUACI()
    
    
    return 0
testCPA()


# In[47]:


#Occlusion-attack analysis
#Robustness against noise
from skimage import io
import skimage
def addPepperNoise(img):
    # add noise test
    Noise_Img = img
    rows,cols=img.shape
    for i in range(2000):
        x=np.random.randint(0,rows)
        y=np.random.randint(0,cols)
        Noise_Img[x,y]=255
    return Noise_Img

def addSaltNoise(img):
    # add noise test
    Noise_Img = img
    rows,cols=img.shape
    for i in range(600):
        x=np.random.randint(0,rows)
        y=np.random.randint(0,cols)
        Noise_Img[x,y]=0
    return Noise_Img

def addSaltAndPepperNoise(img,pro):
    # add noise test
    Noise_Img = img
    rows,cols=img.shape
    iNum = math.floor(rows*cols*pro)
    for i in range(iNum):
        x=np.random.randint(0,rows)
        y=np.random.randint(0,cols)
        if np.random.random_sample()<=0.5:
            Noise_Img[x,y]=0# salt
        else:
            Noise_Img[x,y]=255# Pepper   
    return Noise_Img

def AddGaussianNoise(img,pro):
    Noise_Img = img / 255.00
    
    rows,cols=img.shape
    iNum = math.floor(rows*cols*pro)
    GaussianList = np.random.normal(0, 0.1, (rows,cols))
    for m in range(iNum):
        i=np.random.randint(0,rows)
        j=np.random.randint(0,cols)
        Noise_Img[i][j] = Noise_Img[i][j] + GaussianList[i][j]
        if Noise_Img[i][j] < 0:
            Noise_Img[i][j] = 0
        if Noise_Img[i][j] > 1:
            Noise_Img[i][j] = 1
    Noise_Img = skimage.util.img_as_ubyte(Noise_Img)
    return Noise_Img

def psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

plt.figure(figsize=(35,48),dpi=80)
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B]


picPath = "./test/5.1.12.tiff"
img = np.array(Image.open(picPath).convert('L'))
Img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSaltAndPepperNoise(Img_cipher,0.01)#1%
NoiseImg_plain = DoDec(NoiseImg_cipher,key)  
plt.subplot(4,3,1),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("a. Dec Image (1% Salt&Pepper Noise)")

img = np.array(Image.open(picPath).convert('L'))
print("a. Dec Image (1% Salt&Pepper Noise) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
Img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSaltAndPepperNoise(Img_cipher,0.005)#1%
NoiseImg_plain = DoDec(NoiseImg_cipher,key)  
plt.subplot(4,3,2),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("b. Dec Image (0.5% Salt&Pepper Noise)")

img = np.array(Image.open(picPath).convert('L'))
print("b. Dec Image (0.5% Salt&Pepper Noise) psnr:",psnr(img,NoiseImg_plain))


img = np.array(Image.open(picPath).convert('L'))
Img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSaltAndPepperNoise(Img_cipher,0.001)#1%
NoiseImg_plain = DoDec(NoiseImg_cipher,key)  
plt.subplot(4,3,3),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("c. Dec Image (0.1% Salt&Pepper Noise)")

img = np.array(Image.open(picPath).convert('L'))
print("c. Dec Image (0.1% Salt&Pepper Noise) psnr:",psnr(img,NoiseImg_plain))



img = np.array(Image.open(picPath).convert('L'))
Img_cipher = DoEnc(img,key)
NoiseImg_cipher = AddGaussianNoise(Img_cipher,0.01)#1%
NoiseImg_plain = DoDec(NoiseImg_cipher,key)  
plt.subplot(4,3,4),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("d. Dec Image (1% Gaussian Noise)")

img = np.array(Image.open(picPath).convert('L'))
print("d. Dec Image (1% Gaussian Noise) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
Img_cipher = DoEnc(img,key)
NoiseImg_cipher = AddGaussianNoise(Img_cipher,0.005)#1%
NoiseImg_plain = DoDec(NoiseImg_cipher,key)  
plt.subplot(4,3,5),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("e. Dec Image (0.5% Gaussian Noise)")

img = np.array(Image.open(picPath).convert('L'))
print("e. Dec Image (0.5% Gaussian Noise) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
Img_cipher = DoEnc(img,key)
NoiseImg_cipher = AddGaussianNoise(Img_cipher,0.001)#1%
NoiseImg_plain = DoDec(NoiseImg_cipher,key)  
plt.subplot(4,3,6),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("f. Dec Image (0.1% Gaussian Noise)")

img = np.array(Image.open(picPath).convert('L'))
print("f. Dec Image (0.1% Gaussian Noise) psnr:",psnr(img,NoiseImg_plain))





  



# In[48]:


#Occlusion-attack analysis
#Robustness against data loss
def addSquareLoss(img,ratio,position):
    a,b = img.shape # 256 * 256 
    #ratio : 1/64  1/32 1/16 1/8 
    iRow = 0
    iCol = 0
    if ratio == "1/64":
        iRow = 32
        iCol = 32
    elif ratio == "1/32":
        iRow = 32
        iCol = 64
    elif ratio == "1/16":
        iRow = 64
        iCol = 64
    elif ratio == "1/8":
        iRow = 64
        iCol = 128
    if position == "upper left":
        for i in range(iRow):
            for j in range(iCol):
                img[i][j] = 0
    elif position == "center":
        for i in range(iRow):
            for j in range(iCol):
                img[128-iRow//2+i][128-iCol//2+j] = 0
    elif position  == "lower right":
        for i in range(iRow):
            for j in range(iCol):
                img[255-iRow+i][255-iCol+j] = 0
        
    return img

def psnr(img1, img2):
    mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0**2/mse)

plt.figure(figsize=(35,74),dpi=80)
plt.rcParams['axes.titlesize'] = 32
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B]



picPath = "./test/5.1.12.tiff"
img = np.array(Image.open(picPath).convert('L'))
print("the image's size is :", img.shape)
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/16","upper left")
plt.subplot(6,3,1),plt.imshow(NoiseImg_cipher,cmap="gray")
plt.title("a. Enc Image (1/16, upper left)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
plt.subplot(6,3,4),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("d. Dec image (1/16, upper left)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/16, upper left) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/16","center")
plt.subplot(6,3,2),plt.imshow(NoiseImg_cipher,cmap="gray")
plt.title("b. Enc Image (1/16, center)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
plt.subplot(6,3,5),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("e. Dec image (1/16, center)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/16, center) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/16","lower right")
plt.subplot(6,3,3),plt.imshow(NoiseImg_cipher,cmap="gray")
plt.title("c. Enc Image (1/16, lower right)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
plt.subplot(6,3,6),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("f. Dec image (1/16, lower right)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/16, lower right) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
print("the image's size is :", img.shape)
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/32","upper left")
#plt.subplot(6,3,7),plt.imshow(NoiseImg_cipher,cmap="gray")
#plt.title("g. Enc Image (1/32, upper left)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
#plt.subplot(6,3,10),plt.imshow(NoiseImg_plain,cmap="gray")
#plt.title("j. Dec image (1/32, upper left)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/32, upper left) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/32","center")
#plt.subplot(6,3,8),plt.imshow(NoiseImg_cipher,cmap="gray")
#plt.title("h. Enc Image (1/32, center)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
#plt.subplot(6,3,11),plt.imshow(NoiseImg_plain,cmap="gray")
#plt.title("k. Dec image (1/32, center)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/32, center) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/32","lower right")
#plt.subplot(6,3,9),plt.imshow(NoiseImg_cipher,cmap="gray")
#plt.title("i. Enc Image (1/32, lower right)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
#plt.subplot(6,3,12),plt.imshow(NoiseImg_plain,cmap="gray")
#plt.title("l. Dec image (1/32, lower right)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/32, lower right) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
print("the image's size is :", img.shape)
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/64","upper left")
plt.subplot(6,3,7),plt.imshow(NoiseImg_cipher,cmap="gray")
plt.title("g. Enc Image (1/64, upper left)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
plt.subplot(6,3,10),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("j. Dec image (1/64, upper left)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/64, upper left) psnr:",psnr(img,NoiseImg_plain))

img = np.array(Image.open(picPath).convert('L'))
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/64","center")
plt.subplot(6,3,8),plt.imshow(NoiseImg_cipher,cmap="gray")
plt.title("h. Enc Image (1/64, center)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
plt.subplot(6,3,11),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("k. Dec image (1/64, center)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/64, center) psnr:",psnr(img,NoiseImg_plain))


img = np.array(Image.open(picPath).convert('L'))
img_cipher = DoEnc(img,key)
NoiseImg_cipher = addSquareLoss(img_cipher,"1/64","lower right")
plt.subplot(6,3,9),plt.imshow(NoiseImg_cipher,cmap="gray")
plt.title("i. Enc Image (1/64, lower right)")

NoiseImg_plain = DoDec(NoiseImg_cipher,key)    
plt.subplot(6,3,12),plt.imshow(NoiseImg_plain,cmap="gray")
plt.title("l. Dec image (1/64, lower right)")

img = np.array(Image.open(picPath).convert('L'))
print("Dec image (1/64, lower right) psnr:",psnr(img,NoiseImg_plain))



# In[49]:


#color image encryption and decryption
def DoColorEnc(img,k):
    n = 2 # Encryption round
    #step1. initialization
    
    w,h,d = img.shape #return row and column
    img = img.reshape((w,h*d))
    print(img.shape)
    for i in range(n):
        x0,y0,p1,z0,p2 = getParas(k[i])
        sort,valuekey,x0,y0,z0 = getPRNG(w,h*d,1,x0,y0,p1,z0,p2)
        arr = np.array(sort)
        sortkey = arr[:,0]
        img = Enc(img,sortkey,valuekey,256,2)
    img = img.reshape((w,h,d))
    return img

def DoColorDec(img,k):
    n = 2 # Encryption round    
    w,h,d = img.shape #return row and column
    img = img.reshape((w,h*d))
    for i in range(n-1,-1,-1):
        x0,y0,p1,z0,p2 = getParas(k[i])
        sort,valuekey,x0,y0,z0 = getPRNG(w,h*d,1,x0,y0,p1,z0,p2)
        arr = np.array(sort)
        sortkey = arr[:,0]
        img = Dec(img,sortkey,valuekey,256,2)
    img = img.reshape((w,h,d))
    return img

def ColorTest():
    key = [0x7833A013F4DB0018F4FB4031E9F680BC614A,0x7833A013F4DB0018F4FB4031E9F680BC614B]
    
    icount = 0
    index = 0
    path = './ColorTest/'
    pictures = os.listdir(path)
    for picName in pictures:
        index += 1

    plt.figure(figsize=(26,9*index),dpi=80)
    plt.rcParams['axes.titlesize'] = 25
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    for picName in pictures:
        picPath = path + picName
        lenna_img = np.array(Image.open(picPath))
        print(picName,"'s size is :", lenna_img.shape)    
        plt.subplot(index,3,1+icount*3),plt.imshow(lenna_img,cmap="gray")
        plt.title(picName)
        
        time_start=time.time()
        lenna_cipher = DoColorEnc(lenna_img,key)
        time_end=time.time()    
        plt.subplot(index,3,2+icount*3),plt.imshow(lenna_cipher,cmap="gray")
        plt.title("Enc " + picName)
        
        #Histogram variance
        arr=lenna_cipher[:,:,0].flatten() 
        varValue = Variance(arr,arr.size)    
        print("Red channel's variance is ",str(varValue))
        arr=lenna_cipher[:,:,1].flatten() 
        varValue = Variance(arr,arr.size)    
        print("Green channel's variance is ",str(varValue))
        arr=lenna_cipher[:,:,2].flatten() 
        varValue = Variance(arr,arr.size)    
        print("Blue channel's variance is ",str(varValue))
        
        #Correlation
        print("Correlation")
        print("R channel: Horizontal             Vertical            Diagonal")
        x1=calc(lenna_cipher[:,:,0],'Horizontal')
        x2=calc(lenna_cipher[:,:,0],'Vertical')
        x3=calc(lenna_cipher[:,:,0],'Diagonal')
        print(x1,x2,x3)
        print("G channel: Horizontal             Vertical            Diagonal")
        x1=calc(lenna_cipher[:,:,1],'Horizontal')
        x2=calc(lenna_cipher[:,:,1],'Vertical')
        x3=calc(lenna_cipher[:,:,1],'Diagonal')
        print(x1,x2,x3)
        print("B channel: Horizontal             Vertical            Diagonal")
        x1=calc(lenna_cipher[:,:,2],'Horizontal')
        x2=calc(lenna_cipher[:,:,2],'Vertical')
        x3=calc(lenna_cipher[:,:,2],'Diagonal')
        print(x1,x2,x3)
        
        time_DecStart=time.time()
        lenna_dec = DoColorDec(lenna_cipher,key)
        time_DecEnd=time.time()
        
        plt.subplot(index,3,3+icount*3),plt.imshow(lenna_dec,cmap="gray")
        plt.title("Dec " + picName)
    
        print("Function Enc cost time:", time_end-time_start)
        print("Function Dec cost time:", time_DecEnd-time_DecStart)
        icount += 1   
    return 0

ColorTest()    


# In[ ]:





# In[ ]:




