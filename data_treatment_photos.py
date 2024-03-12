# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:00:06 2024

@author: icarus
"""
import matplotlib.pyplot as plt
from matplotlib import rcParams
from skimage import data, io, filters, exposure, morphology, color
from scipy import ndimage as ndi
from skimage import data, color, draw
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.exposure import histogram
import pandas as pd

import statistics
from scipy.stats import skew
from statistics import NormalDist
from scipy.stats import norm
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import os
black, red, yellow, green, blue, pink = '#343E3D', '#FF5252', '#FFCE54', '#38E4AE', '#51B9FF', '#FB91FF'


rcParams['axes.labelpad'] = 15
plt.rcParams['font.size'] = 12

#%%
def rotate_images(degree):
    outPath = os.getcwd() + "\\rotated"
    path = os.getcwd() + "\\images"

    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the file
        input_path = os.path.join(path, image_path)

        img = Image.open(input_path)

        # rotate the image
        rotated = img.rotate(degree)

        # create full output path, 'example.jpg' 
        # becomes 'rotate_example.jpg', save the file to disk
        fullpath = os.path.join(outPath, 'rotated_'+image_path)
        rotated.save(fullpath)
        
# rotate_images(270)
#%%
def crop_images(x1,y1,x2,y2):
    
    outPath = os.getcwd() + "\\cropped"
    path = os.getcwd() + "\\rotated"

    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the file
        input_path = os.path.join(path, image_path)

        img = Image.open(input_path)

        # crop the image
        cropped = img.crop((x1,y1,x2,y2))

        # create full output path, 'example.jpg' 
        # becomes 'rotate_example.jpg', save the file to disk
        fullpath = os.path.join(outPath, 'cropped_'+image_path)
        cropped.save(fullpath)

# crop_images(1200, 650, 3200, 2650)

#%%
path = os.getcwd()

def gray_images(path):
    
    outPath = os.getcwd() + "\\grayed"
    os.mkdir(outPath)

    # iterate through the names of contents of the folder
    for image_path in os.listdir(path):

        # create the full input path and read the file
        input_path = os.path.join(path, image_path)

        img = Image.open(input_path)

        # grey the image
        grayed = img.convert("L")

        # create full output path, 'example.jpg' 
        # becomes 'rotate_example.jpg', save the file to disk
        fullpath = os.path.join(outPath, 'gray_'+image_path)
        grayed.save(fullpath)
        
gray_images(path)

    #%%

allfiles = os.listdir(os.getcwd())
files = []

for i in allfiles:
    
    each = int(i[18:-4])
    
    files.append(each)
    
files = np.array(files)

files.sort()
files   
#%%

order = np.linspace(1,80,80)

#%%
input_path = os.getcwd() + f"\\{allfiles[3]}"

f"gray_IMG_20240226_{files}.png"
#%% Reading a single image and detecting each circle within it

# input_path = os.getcwd() + "\\imagem_total.png"

def detect_circles(input_path, n=95, sigma=2, min_xdistance=50, min_ydistance=50, minradii=120, maxradii=140, radiivar=6):

    img = io.imread(input_path, as_gray=True)
    img_ = img
    
    edges = canny(img, sigma=sigma)
    
    hough_radii = np.arange(minradii, maxradii, radiivar)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent n circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=n, min_xdistance=min_xdistance, min_ydistance=min_ydistance)
    print(cx, cy)
    
    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    img = color.gray2rgb(img)
    
    
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=img.shape)
        img[circy, circx] = (220, 20, 20)
    
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()
    
    return img, img_, cx, cy, radii, accums


for i in np.arange(0,5,1):
    
    input_path = os.getcwd() + f"\\gray_IMG_20240301_{files[i]}.png"
    img, img_, cx, cy, radii, accums = detect_circles(input_path, n=1, minradii=470, maxradii=700, min_xdistance=500, min_ydistance=500, radiivar=300)
    
    df = img_analysis(img_, cx, cy, radii, check_mask = True)
    
    # df = pd.concat([df, img_analysis(img_, cx, cy, radii)], ignore_index=True)
    
#%%

df = df[1:].reset_index()
# df.to_csv('data1data.csv')
#%%
for i in np.arange(0,80,1):
    print(f"\\gray_IMG_20240226_{files[i]}.png")

# input_path = os.getcwd() + "\\imagem_total.png"

# img, img_, cx, cy, radii, accums = detect_circles(input_path)

#%% FOR CHECKING MASK ONLY

def print_mask(cx, cy, radii, img):

    mask_circle = np.zeros(img.shape[:2], dtype=bool)
    
    for i in range(len(cx)):
        
        print(i)
    
        rr, cc = draw.disk((cy[i], cx[i]), radii[i], shape=mask_circle.shape)
        mask_circle[rr, cc] = 1  
        
    masked = img*mask_circle
    
    # For printing separately
    plt.figure(0)
    
    # For checking mask
    plt.imshow(masked, cmap=plt.cm.gray)
    
print_mask(cx, cy, radii, img_)
#%% Function for analyzing a figure and saving a df with x,y,radii,histogram,rawdata

def img_analysis(img, cx, cy, radii, check_mask = False):

    df = pd.DataFrame(columns=[0,1,2,3,4,5])
    
    for i in range(len(cx)):
        
        print(i)
    
        mask_circle = np.zeros(img.shape[:2], dtype=bool)
        rr, cc = draw.disk((cy[i], cx[i]), radii[i], shape=mask_circle.shape)
        mask_circle[rr, cc] = 1
        
        # For printing separately
        # fig = plt.figure()
        
        # For checking mask
        if check_mask ==  True:
            plt.imshow(img_*mask_circle, cmap=plt.cm.gray)
        
        plt.figure(0)
        
        raw = img*mask_circle
        
        hist = histogram(raw)
        # raw_ravel = raw.ravel()
        
        df.loc[i, 0] = cx[i]
        df.loc[i, 1] = cy[i]
        df.loc[i, 2] = radii[i]
        df.loc[i, 3] = np.array2string(hist[0][1:])
        # df.loc[i, 4] = np.array2string(raw_ravel)
        
        plt.plot(hist[0][1:], alpha=0.5)
    
    df.columns = ['x','y','r','hist','means','sds']

    return df

# df = img_analysis(img_, cx, cy, radii)

#%%

a = df['hist'][0]

i=0

data =  np.array([int(value) for value in df[i][1:-1].split()])


#%%

data = np.zeros(255)

fromdf = np.array([int(value) for value in df['hist'][0][1:-1].split()])

data[:fromdf.shape[0]] = fromdf
#%%
#Calculate means and sd distributions by resampling

def resampler(df, from_str=True):

    means = []
    sds = []
    
    for i in range(len(df)):
        
        #Read i histogram
        
        if from_str == True:
            
            data = np.zeros(255)
            
            fromdf = np.array([int(value) for value in df[i][1:-1].split()])
            print(fromdf)
            
            data[:fromdf.shape[0]] = fromdf
            # 
            print('=='*50)
            print(len(data))
            print(data)
            print(type(data))
            print('=='*50)
            # size = len(data)
        #Calculate Moving Average for resampling 
        
        #MA = pd.Series(data).rolling(window=30, center=True).mean()
        
        #Generate space and resamples histogram
        
        x = np.linspace(0,255,256)
        resamples = np.random.choice((x[:-1] + x[1:])/2, size=100*5, p=data/data.sum())
        
        #Calculates normal distribution, mean and sd
        
        normal_dist = NormalDist.from_samples(resamples)
        mean, sd = normal_dist.mean, normal_dist.stdev
        
        #Store values
        
        means.append(mean)
        sds.append(sd)
        
        #Plot for verification
        
        plt.figure(0)
        
        plt.hist(resamples,alpha=0.5, color=blue)
        plt.plot(data/data.sum()*5000, alpha=0.5, color=black)
        plt.plot(x, norm.pdf(x, mean, sd)*5000, color=red)
    
        plt.figure(1)
        
        plt.plot(x, norm.pdf(x, mean, sd)*5000, color=red, alpha=0.5)
        
        plt.xlabel('Grayscale')
        plt.ylabel('Frequency')
        # plt.plot(x, norm.pdf(x, mean_all, sd_all)*5000, color=black, alpha=1) #For average distribution
        
    means = np.array(means)
    sds = np.array(sds)
    
    return means, sds

# df = pd.read_csv('data1data.csv')

means, sds = resampler(df['hist'])

#%%

df['means'], df['sds'] = means, sds

# df.to_csv('data1_1.csv', index=False)

#%% smooth histograms

df = pd.read_csv('data.csv')

#%%

x = np.linspace(0,255,100)

mean_all = means.mean()
sd_all = means.std(ddof=1)

plt.plot(x, norm.pdf(x, mean_all, sd_all)*5000, color=black, alpha=1)
plt.xlabel('Grey scale')
plt.ylabel('Frequency')

#%%

median_all = np.median(means)
sds_true_all = sds.mean()

df['means_normalized'] = means - mean_all
df['sds_true_normalized'] = sds - sds_true_all

plt.hist(sds)
#%%

def normalize(data):
    
    norm_data = (data - data.min()) / (data.max() - data.min())
    
    return norm_data

def correction_factor(data1, data2):
    
    cf = data2.mean()/data1.mean()
    
    return cf
#%%

df = pd.read_csv('data1_1.csv')
df2 = pd.read_csv('data2.csv')

norm1 = normalize(df.means)
norm2 = normalize(df2.means)

cf = correction_factor(norm1, norm2)

# plt.scatter(np.linspace(1,80,len(df.means)), norm1)
plt.scatter(np.linspace(1,80,len(df2.means)), norm2)
plt.ylim([0,1])
#%% DATA TO COLOR

def data_to_color(df, data='means', savefig=False, dfname='df', cf=1, colormap="Reds", title=False):
    
    plt.figure()
    
    normalized = normalize(df[data]*cf)
    
    plt.scatter(df.x, df.y, s=130, c=normalized, label='colors', cmap=colormap)
    
    plt.xticks([])
    plt.yticks([])
    
    if data == 'means':
        plt.title('Normalized $\mu$', pad=15)
    # elif data == 'sds':
    #     plt.title('Normalized $\sigma$', pad=15)
    elif data == 'sdstotal':
        plt.title('Normalized $\sigma $', pad=15)
    elif title != False:
        plt.title(f'{title}', pad=15)
        
    cbar = plt.colorbar()
    # cbar.set_label('# of contacts', rotation=90)
    
    if savefig == True:
        plt.savefig(f'{dfname}_normalized_{data}.png', dpi=300, bbox_inches='tight')
        
    return 
#%%
df3=pd.DataFrame()
df3['means'] = (df['means']+df2['means'])/2
df3['sds'] = (((df3['means'] - df3['means'].mean())**2)/len(df3['means']))**0.5  
df3['sdstotal'] = (df['sds']+df2['sds'])/2
df3['x'] = df['x']
df3['y'] = df['y']
df3['allinone'] = normalize(df3['sdstotal'])+normalize(df3['sds'])
#%%
# data_to_color(df, data='means', cf=1)
# data_to_color(df2, data='means', cf=1)
data_to_color(df3, data='means', cf=1, colormap='Reds', savefig=True)
data_to_color(df3, data='sds', cf=1, colormap='Reds', title="Normalized $\sigma_{_\mu}$", savefig=True)
#%%
# data_to_color(df, data='sds', cf=1)
# data_to_color(df2, data='sds', cf=1)
# data_to_color(df3, data='sds', cf=1, colormap='Blues')
#%%
# data_to_color(df, data='sdstotal', cf=1)
# data_to_color(df2, data='sdstotal', cf=1)
data_to_color(df3, data='sdstotal', cf=1, colormap='Blues', savefig=True)

#%%

data_to_color(df3, data='allinone', cf=1, colormap='Purples', title='All in one', savefig=True)

#%%

x = range(80)

u1 = df['means']
u2 = df2['means']

plt.scatter(x, u1, color=blue)
plt.scatter(x, u2, color=red)

plt.xlabel('Order')
plt.ylabel('Grayscale')
plt.ylim([0,255])
plt.ylim([80,170])


##################################################
#%%

plt.hist(normalize(df3.sds))
#%%

plt.hist(normalize(df3.sdstotal))
#%% BY "TIME"

x = np.arange(1,len(norm1)+len(norm2)+1,1)

concat = pd.concat([norm1, norm2]).reset_index(drop=True)

plt.scatter(x, concat, color=black)

plt.ylim([-.05, 1.05])

#%% BY "TIME" but COLUMNS COLORED

columns = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,5,5,5,5,5,5,6,6,6,7,7,7,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12,12,13,13,13,14,14,14,15, 16, 16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,20,20,20,20,20,21,21,21,21,21,21,22,22,22,23,23,23,24,25,25,25,25,25,25,25,25,25,26,26,26,26,26,27,27,27,27,27,27,28,28,28,29,29,29,30]
colors=plt.get_cmap('rainbow', 31)

columns = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,5,5,5,5,5,5,6,6,6,7,7,7,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12,12,13,13,13,14,14,14,15, 0, 0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,7,7,7,8,9,9,9,9,9,9,9,9,9,10,10,10,10,10,11,11,11,11,11,11,12,12,12,13,13,13,14]
colors=plt.get_cmap('rainbow', 16)

x = np.arange(0,len(concat),1)

for i in np.arange(0,len(concat),1):

    # plt.scatter(df.pos, df.means, color=black)
    plt.scatter(x[i], concat[i], color=colors(columns[i]))
    
    plt.xlabel('Sequence')
    plt.ylabel('SDs')
    
    # plt.axhline(df.means.mean(),color=red,alpha=0.5)
    
    # plt.ylim(0,255)
    
plt.axhline(concat.mean(), color=black,alpha=0.5)
#%%

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(df.x, df.y, means, cmap='viridis')


#%%

import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

fig = px.scatter_3d(df, x='x', y='y', z='means')
fig.show()

#%%

df = pd.read_csv('data2.csv')

fig, ax = plt.subplots()
ax.scatter(df.x, df.y, alpha=0.2)

for i in range(80):
    plt.annotate(df.pos2[i], (df.x[i], df.y[i]), size=7)


#%% check numbering

for i in range(80):
    df2 = df.iloc[i]
    
    plt.figure()
    
    plt.title(i)
    plt.scatter(df.x, df.y, alpha=0.2)
    plt.scatter(df2.x, df2.y, color=red)

#%%

columns = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,5,5,5,5,5,5,6,6,6,7,7,7,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12,12,13,13,13,14,14,14,15]

# normalizado = (df.means - df['means'].min()) / (df['means'].max() - df['means'].min())

# plt.scatter(df.pos, df.means, color=black)
plt.scatter(df.pos, normalizado, color=black)

plt.xlabel('Sequence')
plt.ylabel('Mean')

# plt.axhline(df.means.mean(),color=red,alpha=0.5)

# plt.ylim(0,255)

plt.axhline(normalizado.mean(), color=red,alpha=0.5)

#%% color by columns
#12, 7

columns = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,5,5,5,5,5,5,6,6,6,7,7,7,8,9,9,9,9,9,9,9,10,10,10,10,10,10,10,10,10,11,11,11,11,11,12,12,12,12,12,12,13,13,13,14,14,14,15, 16, 16,16,16,16,16,16,16,16,16,16,17,17,17,17,17,17,17,18,18,18,18,18,18,18,19,19,19,19,19,19,19,19,19,20,20,20,20,20,21,21,21,21,21,21,22,22,22,23,23,23,24,25,25,25,25,25,25,25,25,25,26,26,26,26,26,27,27,27,27,27,27,28,28,28,29,29,29,30]
colors=plt.get_cmap('rainbow', 31)

# normalizado = (df.means - df['means'].min()) / (df['means'].max() - df['means'].min())

for i in range(80):

    # plt.scatter(df.pos, df.means, color=black)
    plt.scatter(df.pos[i], [i], color=colors(columns[i]))
    
    plt.xlabel('Sequence')
    plt.ylabel('SDs')
    
    # plt.axhline(df.means.mean(),color=red,alpha=0.5)
    
    # plt.ylim(0,255)
    
plt.axhline(normalizado.mean(), color=black,alpha=0.5)


#%%
colunas = [np.arange(0,13), np.arange(13,20), np.arange(20,29), np.arange(29,34), np.arange(34,40), np.arange(40,43), np.arange(43,46), np.arange(46,47), np.arange(47,54), np.arange(54,63), np.arange(63,68), np.arange(68,74), np.arange(74,77), np.arange(77,80), np.arange(80,81), np.arange(81,81+13)]

#%%
for i in colunas:
    
    print(i.min())
    print(i.max()+1)
#%%
counter = 1

for i in colunas:
    
    
    
    fig = plt.figure()
    print(i)
    
    plt.scatter(i, concat[i.min(): i.max()+1], color=black)
    
    plt.title(f'coluna {counter}')
    plt.axhline(concat.mean(), color=black,alpha=0.5)
    plt.ylim([0,.6])

    counter += 1



#%% Convolution

distributions = 1

def convolve_distributions(distributions):
    result = distributions[0]
    for dist in distributions[1:]:
        result = np.convolve(result, dist, mode='full')
    return result


result_dist = convolve_distributions(distributions)
#%%
from scipy.stats import norm

x = np.linspace(0,255,100)

df = pd.read_csv('data2.csv')

data = read_str(df['hist'][4])
#%%

#Generate space and resamples histogram

x = np.linspace(0,255,256)
resamples = np.random.choice((x[:-1] + x[1:])/2, size=100*5, p=data/data.sum())

#Calculates normal distribution, mean and sd

normal_dist = NormalDist.from_samples(resamples)
mean, sd = normal_dist.mean, normal_dist.stdev

#Plot for verification

plt.hist(resamples,alpha=0.5, color=blue)
plt.plot(data/data.sum()*5000, alpha=0.5, color=black)
plt.plot(x, norm.pdf(x, mean, sd)*5000, color=red)

#%%
# MA = pd.Series(data).rolling(window=30, center=True).mean().dropna().reset_index(drop=True)

# plt.figure(0)
# x = np.linspace(0,255,256)
# # plt.figure()
# # plt.plot(data, alpha=0.2)
# plt.ylim([0,2000])
# plt.xlim([50,250])
# plt.plot(MA, alpha=0.5)




# plt.figure(1)
plt.plot(x, norm.pdf(x, mean, sd))
plt.plot(data/data.sum())


#%%

##########Alternative method for processing each image, one at a time,


# outPath = os.getcwd() + "\\grayed"
path = os.getcwd() + "\\grayed"

# iterate through the names of contents of the folder
for image_path in os.listdir(path):
    
    input_path = os.path.join(path, image_path)

    img = io.imread(input_path, as_gray=True)
    img_ = img
    
    edges = canny(img, sigma=2)
    
    hough_radii = np.arange(130,133, 6)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent n circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=15)
    print(cx, cy)
    
    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    img = color.gray2rgb(img)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius,
                                        shape=img.shape)
        img[circy, circx] = (220, 20, 20)
    
    ax.imshow(img, cmap=plt.cm.gray)
    plt.show()
    
    for i in range(len(cx)):
        
        print(i)
            
        mask_circle = np.zeros(img.shape[:2], dtype=bool)
        rr, cc = draw.disk((cy[i], cx[i]), radii[i], shape=mask_circle.shape)
        mask_circle[rr, cc] = 1
        
        plt.figure(0)
        
        plt.imshow(img_*mask_circle, cmap=plt.cm.gray)
        
        plt.figure(1)
        
        hist = histogram(img_*mask_circle)
        plt.plot(hist[0][1:])
        
        print(cx[i])

