





import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from tqdm.auto import tqdm as tq
import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils

import numpy as np
import pandas as pd
from math import sqrt, acos, pi, sin, cos
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import average_precision_score
from multiprocessing import Pool

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import gc

from torch.utils.tensorboard.writer import SummaryWriter
from time import clock_gettime, CLOCK_MONOTONIC
from tqdm.notebook import tqdm_notebook as tqdm

import albumentations as A


PATH = '/disks/ssd/pku-autonomous-driving/'
CPFOLDER = '/disks/hdd/pku/article/quaternion'
writer = SummaryWriter(CPFOLDER)
os.listdir(PATH)





os.environ['CUDA_VISIBLE_DEVICES'] = '0'






SWITCH_LOSS_EPOCH = 5
print(torch.__version__)





from  keras.applications.inception_resnet_v2 import InceptionResNetV2





train = pd.read_csv(PATH + 'train.csv')
test = pd.read_csv(PATH + 'sample_submission.csv')
bad_list = ['ID_1a5a10365',
'ID_1db0533c7',
'ID_53c3fe91a',
'ID_408f58e9f',
'ID_4445ae041',
'ID_bb1d991f6',
'ID_c44983aeb',
'ID_f30ebe4d4']
train = train.loc[~train['ImageId'].isin(bad_list)]

camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)





def imread(path, fast_mode=False):
    img = cv2.imread(path)
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

img = imread(PATH + 'train_images/ID_8a6e65317' + '.jpg')
IMG_SHAPE = img.shape





def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def coords2str(coords, names):
    s = []
    for c in coords:
        for n in names:
            s.append(str(c.get(n, 0)))
    return ' '.join(s)





def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x





def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image
        ys: y coordinates in the image
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] 
    return img_xs, img_ys





from math import sin, cos


def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def Rot_to_euler(Rt):
    sin_p = -Rt[1, 2]
    p_1 = np.arcsin(sin_p)
    p_2 = np.pi - p_1
    ypr = []
    if np.abs(np.cos(p_1)) < 1e-5:
        r = 0
        if np.abs(Rt[1, 2] + 1) < 1e-5:
            y = np.arctan2(Rt[0, 1], Rt[0, 0])
        else:
            y = np.arctan2(-Rt[0, 1], Rt[0, 0])
        ypr.append((y, p, r))
    else:
        for p in [p_1, p_2]:
            cos_p = np.cos(p)
            r = np.arctan2(Rt[1, 0] / cos_p, Rt[1, 1] / cos_p)
            y = np.arctan2(Rt[0, 2] / cos_p, Rt[2, 2] / cos_p)
            ypr.append((y, p, r))
    if len(ypr) == 1:
        return ypr[0]
    else:
        if np.abs(ypr[0][1]) < np.abs(ypr[1][1]):
            return ypr[0]
        else:
            return ypr[1]





def get_rotation_matrix(u, v):
    x, y, _ = camera_matrix_inv.dot(np.array([u, v, 1]))
    alpha_x = np.arctan(y)
    alpha_y = -np.arctan(x)
    
    Rx = R.from_rotvec(alpha_x * np.array([1, 0, 0])).as_dcm()
    Ry = R.from_rotvec(alpha_y * np.array([0, 1, 0])).as_dcm()

    return Rx.dot(Ry)

def to_relative(u, v, regr_dict):
    A = get_rotation_matrix(u, v)
    
    yaw, pitch, roll = -regr_dict['pitch'], -regr_dict['yaw'], -regr_dict['roll']
    R = euler_to_Rot(yaw, pitch, roll).T
    R_new = A.dot(R)
    yaw, pitch, roll = Rot_to_euler(R_new.T)
    regr_dict['pitch'], regr_dict['yaw'], regr_dict['roll'] = -yaw, -pitch, -roll
    
    return regr_dict

def to_absolute(u, v, regr_dict):
    A = get_rotation_matrix(u, v)
    A = np.linalg.inv(A)
    
    yaw, pitch, roll = -regr_dict['pitch'], -regr_dict['yaw'], -regr_dict['roll']
    R = euler_to_Rot(yaw, pitch, roll).T
    R_new = A.dot(R)
    yaw, pitch, roll = Rot_to_euler(R_new.T)
    regr_dict['pitch'], regr_dict['yaw'], regr_dict['roll'] = -yaw, -pitch, -roll
    
    return regr_dict





def draw_line(image, points):
    color1 = (255, 0, 0)
    color2 = (0, 255, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color1, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color1, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color1, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color2, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(10), (0, 255, 0), -1)
        cv2.putText(image, '{:.2f}'.format(p_z / 2), (p_x + 20, p_y + 20), 0, 1.3, (0, 255, 0), 4)


    return image





def visualize(img, coords):
    
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img





scale = 1.0





IMG_WIDTH = int(scale * 1600)
IMG_HEIGHT = int(scale * 700)
MODEL_SCALE = 8

def _regr_preprocess(u, v, regr_dict):
    regr_dict = to_relative(u, v, regr_dict)
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    



    regr_dict = euler_to_quaternion(regr_dict)
    
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict

def _regr_back(u, v, regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    




    regr_dict = quaternion_to_euler(regr_dict)
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    to_absolute(u, v, regr_dict)
    return regr_dict

def euler_to_quaternion(regr_dict):
    yaw, pitch, roll = -regr_dict['pitch'], -regr_dict['yaw'], -regr_dict['roll']
    Rot = euler_to_Rot(yaw, pitch, roll).T
    q = matrix_to_quaternion(Rot)
    regr_dict['yaw'] = q[0]
    regr_dict['roll'] = q[1]
    regr_dict['pitch_sin'] = q[2]
    regr_dict['pitch_cos'] = q[3]
    return regr_dict

def matrix_to_quaternion(Rot):
    rotation = R.from_dcm(Rot)
    return rotation.as_quat()

def quaternion_to_euler(regr_dict):
    q = np.zeros((4,), dtype=np.float32)
    q[0] = regr_dict['yaw']
    q[1] = regr_dict['roll']
    q[2] = regr_dict['pitch_sin']
    q[3] = regr_dict['pitch_cos']
    Rot = quaternion_to_matrix(q)
    yaw, pitch, roll = Rot_to_euler(Rot.T)
    regr_dict['pitch'], regr_dict['yaw'], regr_dict['roll'] = -yaw, -pitch, -roll
    return regr_dict

def quaternion_to_matrix(q):
    rotation = R.from_quat(q)
    return rotation.as_dcm()

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = 0 * np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 4]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def preprocess_bg(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.zeros_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 4]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    
    if flip:
        img = img[:,::-1]
    return (img).astype('float32')


def draw_msra_gaussian(heatmap, center, sigma=1):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def draw_dense_reg(regmap, heatmap, center, value, radius=1, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma= diameter/6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter*2+1, diameter*2+1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter*2+1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                             radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                      radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: 
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1-idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    
    regr = regr.transpose([2,0,1])
    
    for u, v, regr_dict in zip(xs, ys, coords):
        x = (v - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (u + img.shape[1] // 4) * IMG_WIDTH / (img.shape[1] * 1.5) / MODEL_SCALE
        
        
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask = draw_msra_gaussian(mask, [y,x], 1)
            
            
            
            regr_dict = _regr_preprocess(u, v, regr_dict)
            regrs = [regr_dict[n] for n in sorted(regr_dict)]
            
            
            
            regr = draw_dense_reg(regr, mask, [y,x], regrs, 3, True)
            
    regr = regr.transpose([1,2,0])
    
    
    for i, r in enumerate(regr[0,0,:]):
        
        regr[:,:,i] *= (mask > 0.1)
    
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr





class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None, flag=None, augment=False):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training
        self.flag = flag
        self.augment = augment
        
        self.color_augmentations = A.Compose([
            A.OneOf([
                A.RandomBrightness(limit=0.2, always_apply=True),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=True),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, always_apply=True),
                A.RandomContrast(limit=0.2, always_apply=True)
            ], p=0.7),
            A.OneOf([
                A.Blur(blur_limit=7, always_apply=True),
                A.MedianBlur(blur_limit=7, always_apply=True),
                A.GaussNoise(var_limit=(5, 50), always_apply=True)
            ], p=0.7)
        ])

    def __len__(self):
        if self.flag:
            return len(self.df)
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        
        img0 = imread(img_name, False)
        
        self.A2d = np.eye(3)
        
        if self.augment:
            img0 = self.color_augmentations(image=img0)['image']
            
            if np.random.uniform() < 0.5:
                img0, labels = self.apply_flip(img0, labels)
            
            A = self.get_3d_transform_matrix()
            img0, labels = self.apply_rotation(A, img0, labels)
        
        img = preprocess_image(img0)
        mean = np.array([0.485, 0.456, 0.406], dtype='float32')
        std =  np.array([0.229, 0.224, 0.225], dtype='float32')
        img = (img - mean) / std
        img = np.rollaxis(img, 2, 0)
        
        
        if self.training:
            mask, regr = get_mask_and_regr(img0, labels)
            regr = np.rollaxis(regr, 2, 0)
        else:
            mask, regr = 0, 0
        
        return [img, mask, regr]
    
    def get_3d_transform_matrix(self):
        A = np.eye(3)
        if np.random.uniform() < 0.5:
            alpha = np.pi * np.random.normal(scale=1) / 180
            Rz = R.from_rotvec(alpha * np.array([0, 0, 1])).as_dcm()
            A = Rz.dot(A)
        if np.random.uniform() < 0.5:
            alpha = np.pi * np.random.normal(scale=0.5) / 180
            Rx = R.from_rotvec(alpha * np.array([1, 0, 0])).as_dcm()
            A = Rx.dot(A)
        if np.random.uniform() < 0.5:
            alpha = np.pi * np.random.normal(scale=0.5) / 180
            Ry = R.from_rotvec(alpha * np.array([0, 1, 0])).as_dcm()
            A = Ry.dot(A)
        if np.random.uniform() < 0.5:
            H, W = 2710, 3384
            scale = np.random.uniform(0.95, 1.0)
            h, w = int(scale * H), int(scale * W)
            x = np.random.uniform(0, W - w)
            y = np.random.uniform(0, H - h)
            r1 = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32")
            r2 = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype="float32")
            A2d = cv2.getPerspectiveTransform(r1, r2)
            A = camera_matrix_inv.dot(A2d).dot(camera_matrix).dot(A)
            
        return A
    
    def apply_rotation(self, A, img0, labels):
        H, W = img0.shape[:2]
        A2d = camera_matrix.dot(A).dot(camera_matrix_inv)
        img0 = cv2.warpPerspective(img0, A2d, (W, H))
        coords = str2coords(labels)
        for regr_dict in coords:
            center = np.array([regr_dict['x'], regr_dict['y'], regr_dict['z']])
            center = A.dot(center)
            regr_dict['x'], regr_dict['y'], regr_dict['z'] = center
            yaw, pitch, roll = -regr_dict['pitch'], -regr_dict['yaw'], -regr_dict['roll']
            R = euler_to_Rot(yaw, pitch, roll).T
            R_new = A.dot(R)
            yaw, pitch, roll = Rot_to_euler(R_new.T)
            regr_dict['pitch'], regr_dict['yaw'], regr_dict['roll'] = -yaw, -pitch, -roll
        labels = coords2str(coords, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z'])
        self.A2d = A2d.dot(self.A2d)
        return img0, labels
    
    def apply_flip(self, img0, labels):
        H, W = img0.shape[:2]
        A = np.diag([-1, 1, 1])
        A2d = camera_matrix.dot(A).dot(camera_matrix_inv)
        img0 = cv2.warpPerspective(img0, A2d, (W, H), borderMode=cv2.BORDER_REPLICATE)
        coords = str2coords(labels)
        for regr_dict in coords:
            center = np.array([regr_dict['x'], regr_dict['y'], regr_dict['z']])
            center = A.dot(center)
            regr_dict['x'], regr_dict['y'], regr_dict['z'] = center
            yaw, pitch, roll = -regr_dict['pitch'], -regr_dict['yaw'], -regr_dict['roll']
            R = euler_to_Rot(yaw, pitch, roll)
            R_new = A.dot(R)
            yaw, pitch, roll = Rot_to_euler(R_new)
            regr_dict['pitch'], regr_dict['yaw'], regr_dict['roll'] = -yaw, -pitch, -roll
        labels = coords2str(coords, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z'])
        self.A2d = A2d.dot(self.A2d)
        return img0, labels





def extract_coords(prediction):
    logits = prediction[0]
    regr_output = prediction[1:]
    points = np.argwhere(logits > 0)
    asort = np.argsort(logits[points[:, 0], points[:, 1]])
    points = points[asort[::-1]]
    points = points[:min(len(points), 100)]
    coords = []
    if len(points) > 0:
        col_names = sorted(['x', 'y', 'z', 'yaw', 'pitch_sin', 'pitch_cos', 'roll'])
        for r, c in points:
            u = c * MODEL_SCALE * (3384 * 1.5) / IMG_WIDTH - 3384 // 4
            v = r * MODEL_SCALE * (2710 // 2) / IMG_HEIGHT + 2710 // 2
            regr_dict = dict(zip(col_names, regr_output[:, r, c]))
            coords.append(_regr_back(u, v, regr_dict))
            coords[-1]['confidence'] = 1 / (1 + np.exp(-logits[r, c]))
            coords[-1]['x'], coords[-1]['y'], coords[-1]['z'] = optimize_xy(r, c, coords[-1]['x'], coords[-1]['y'], coords[-1]['z'])
        coords = clear_duplicates(coords)
    return coords





train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'

df_train, df_dev = train_test_split(train, test_size=0.1, random_state=63)
df_test = test


train_dataset = CarDataset(df_train, train_images_dir, flag=True, augment=True)
dev_dataset = CarDataset(df_dev, train_images_dir, flag=False)
test_dataset = CarDataset(df_test, test_images_dir, flag=False)





BATCH_SIZE = 1


train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)





class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        
        
        
        
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x

def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh







model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'inceptionresnetv2': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None


    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(16, planes * self.expansion)

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride), 
                nn.GroupNorm(16, planes * self.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
 
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNetFeatures(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNetFeatures, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.GroupNorm(16, 64)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        
        
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        conv1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        conv1 = F.max_pool2d(conv1, 3, stride=2, padding=1)

        feats4 = self.layer1(conv1)
        feats8 = self.layer2(feats4)
        feats16 = self.layer3(feats8)
        feats32 = self.layer4(feats16)

        return feats8, feats16, feats32



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def  resnext50(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnext50_32x4d']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet34']))
    return model

def inceptionresnetv2(pretrained=False, **kwargs):
    """Constructs a inceptionresnetv2.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['inceptionresnetv2']))
    return model

def densenet201(pretrained=False, **kwargs):
    """Constructs a densenet201 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetFeatures(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['densenet201']))
    return model


def _load_pretrained(model, pretrained):
    model_dict = model.state_dict()
    pretrained = {k : v for k, v in pretrained.items() if k in model_dict}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)





base_model = inceptionresnetv2(pretrained=True)
base_model





class CentResnet(nn.Module):
    '''Mixture of previous classes'''
    def __init__(self, n_classes):
        super(CentResnet, self).__init__()
        self.base_model = base_model
        
        
        self.lat8 = nn.Conv2d(128, 256, 1)
        self.lat16 = nn.Conv2d(256, 256, 1)
        self.lat32 = nn.Conv2d(512, 256, 1)
        self.bn8 = nn.GroupNorm(16, 256)
        self.bn16 = nn.GroupNorm(16, 256)
        self.bn32 = nn.GroupNorm(16, 256)

       
        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)
        
        self.mp = nn.MaxPool2d(2)
        
        self.up1 = up(1282 , 512) 
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)
        
    
    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        
        
                
        feats8, feats16, feats32 = self.base_model(x)
        lat8 = F.relu(self.bn8(self.lat8(feats8)))
        lat16 = F.relu(self.bn16(self.lat16(feats16)))
        lat32 = F.relu(self.bn32(self.lat32(feats32)))
        
        
        mesh2 = get_mesh(batch_size, lat32.shape[2], lat32.shape[3])
        feats = torch.cat([lat32, mesh2], 1)
        
        
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x





import math
import torch
from torch.optim.optimizer import Optimizer, required

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                
                if N_sma >= 5:            
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss






device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

patience = 5

model = CentResnet(8).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)






def neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pred = pred.unsqueeze(1).float()
    gt = gt.unsqueeze(1).float()

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss





def criterion(prediction, mask, regr, size_average=True):
    
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = neg_loss(pred_mask, mask)

    
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / (1e-5 + mask.sum(1).sum(1))
    regr_loss = regr_loss.mean(0)

    
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss , regr_loss





DISTANCE_THRESH_CLEAR = 2

def convert_3d_to_2d(x, y, z, fx = 2304.5479, fy = 2305.8757, cx = 1686.2379, cy = 1354.9849):
    
    return x * fx / z + cx, y * fy / z + cy

def optimize_xy(r, c, x0, y0, z0):
    def distance_fn(xyz):
        x, y, z = xyz
        x, y = convert_3d_to_2d(x, y, z0)
        y, x = x, y
        x = (x - IMG_SHAPE[0] // 2) * IMG_HEIGHT / (IMG_SHAPE[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + IMG_SHAPE[1] // 4) * IMG_WIDTH / (IMG_SHAPE[1] * 1.5) / MODEL_SCALE
        y = np.round(y).astype('int')
        return (x-r)**2 + (y-c)**2
    
    res = minimize(distance_fn, [x0, y0, z0], method='Powell')
    x_new, y_new, z_new = res.x
    return x_new, y_new, z0

def clear_duplicates(coords):
    for c1 in coords:
        xyz1 = np.array([c1['x'], c1['y'], c1['z']])
        for c2 in coords:
            xyz2 = np.array([c2['x'], c2['y'], c2['z']])
            distance = np.sqrt(((xyz1 - xyz2)**2).sum())
            if distance < DISTANCE_THRESH_CLEAR:
                if c1['confidence'] < c2['confidence']:
                    c1['confidence'] = -1
    return [c for c in coords if c['confidence'] > 0]





def expand_df(df, PredictionStringCols):
    df = df.dropna().copy()
    df['NumCars'] = [int((x.count(' ')+1)/7) for x in df['PredictionString']]

    image_id_expanded = [item for item, count in zip(df['ImageId'], df['NumCars']) for i in range(count)]
    prediction_strings_expanded = df['PredictionString'].str.split(' ',expand = True).values.reshape(-1,7).astype(float)
    prediction_strings_expanded = prediction_strings_expanded[~np.isnan(prediction_strings_expanded).all(axis=1)]
    df = pd.DataFrame(
        {
            'ImageId': image_id_expanded,
            PredictionStringCols[0]:prediction_strings_expanded[:,0],
            PredictionStringCols[1]:prediction_strings_expanded[:,1],
            PredictionStringCols[2]:prediction_strings_expanded[:,2],
            PredictionStringCols[3]:prediction_strings_expanded[:,3],
            PredictionStringCols[4]:prediction_strings_expanded[:,4],
            PredictionStringCols[5]:prediction_strings_expanded[:,5],
            PredictionStringCols[6]:prediction_strings_expanded[:,6]
        })
    return df

def TranslationDistance(p,g, abs_dist = False):
    dx = p['x'] - g['x']
    dy = p['y'] - g['y']
    dz = p['z'] - g['z']
    diff0 = (g['x']**2 + g['y']**2 + g['z']**2)**0.5
    diff1 = (dx**2 + dy**2 + dz**2)**0.5
    if abs_dist:
        diff = diff1
    else:
        diff = diff1/diff0
    return diff

def RotationDistance(p, g):
    true=[ g['pitch'] ,g['yaw'] ,g['roll'] ]
    pred=[ p['pitch'] ,p['yaw'] ,p['roll'] ]
    q1 = R.from_euler('xyz', true)
    q2 = R.from_euler('xyz', pred)
    diff = R.inv(q2) * q1
    W = np.clip(diff.as_quat()[-1], -1., 1.)
    
    
    
    
    
    
    W = (acos(W)*360)/pi
    if W > 180:
        W = 360 - W
    return W





thres_tr_list = [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
thres_ro_list = [50, 45, 40, 35, 30, 25, 20, 15, 10, 5]

def check_match(idx):
    keep_gt=False
    thre_tr_dist = thres_tr_list[idx]
    thre_ro_dist = thres_ro_list[idx]
    train_dict = {imgID:str2coords(s, names=['carid_or_score', 'pitch', 'yaw', 'roll', 'x', 'y', 'z']) for imgID,s in zip(train_df_['ImageId'],train_df_['PredictionString'])}
    valid_dict = {imgID:str2coords(s, names=['pitch', 'yaw', 'roll', 'x', 'y', 'z', 'carid_or_score']) for imgID,s in zip(valid_df_['ImageId'],valid_df_['PredictionString'])}
    result_flg = [] 
    scores = []
    MAX_VAL = 10**10
    for img_id in valid_dict:
        for pcar in sorted(valid_dict[img_id], key=lambda x: -x['carid_or_score']):
            
            min_tr_dist = MAX_VAL
            min_idx = -1
            for idx, gcar in enumerate(train_dict[img_id]):
                tr_dist = TranslationDistance(pcar,gcar)
                if tr_dist < min_tr_dist:
                    min_tr_dist = tr_dist
                    min_ro_dist = RotationDistance(pcar,gcar)
                    min_idx = idx
                    
            
            if min_tr_dist < thre_tr_dist and min_ro_dist < thre_ro_dist:
                if not keep_gt:
                    train_dict[img_id].pop(min_idx)
                result_flg.append(1)
            else:
                result_flg.append(0)
            scores.append(pcar['carid_or_score'])
    
    return result_flg, scores





train_df_ = None
valid_df_ = None

def calk_map(filename):
    global train_df_, valid_df_
    
    valid_df_ = pd.read_csv(filename, dtype=str)
    expanded_valid_df = expand_df(valid_df_, ['pitch','yaw','roll','x','y','z','Score'])
    valid_df_ = valid_df_.fillna('')

    train_df_ = pd.read_csv('/disks/ssd/pku-autonomous-driving/train.csv')
    train_df_ = train_df_[train_df_.ImageId.isin(valid_df_.ImageId.unique())]
    
    
    
    
    expanded_train_df = expand_df(train_df_, ['model_type','pitch','yaw','roll','x','y','z'])

    max_workers = 10
    n_gt = len(expanded_train_df)
    ap_list = []
    with Pool(processes=max_workers)as p:
        for result_flg, scores in p.imap(check_match, range(10)):
            if np.sum(result_flg) > 0:
                n_tp = np.sum(result_flg)
                recall = n_tp/n_gt
                ap = average_precision_score(result_flg, scores)*recall
            else:
                ap = 0
            ap_list.append(ap)
    return np.mean(ap_list)





def train(epoch, history=None):
    model.train()
    t = tqdm(train_loader)
    train_loss = 0
    train_mask_loss = 0
    train_regr_loss = 0
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(t):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        
        optimizer.zero_grad()
        output = model(img_batch)
        
        loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch)
        
        t.set_description(f'train_loss (l={loss:.3f})(m={mask_loss:.2f}) (r={regr_loss:.4f}')
        
        train_loss += loss
        train_mask_loss += mask_loss
        train_regr_loss += regr_loss
        
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()
        
        optimizer.step()
    
    train_loss /= len(train_loader)
    train_mask_loss /= len(train_loader)
    train_regr_loss /= len(train_loader)

    writer.add_scalar('LR', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/mask_loss', train_mask_loss, epoch)
    writer.add_scalar('train/regr_loss', train_regr_loss, epoch)
    
    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        train_loss,
        train_mask_loss,
        train_regr_loss))

def evaluate(epoch, history=None):
    model.eval()
    val_preds = []
    loss = 0
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in tqdm(dev_loader):
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            loss,mask_loss, regr_loss= criterion(output, mask_batch, regr_batch, size_average=False)
            valid_loss += loss.data
            valid_mask_loss += mask_loss.data
            valid_regr_loss += regr_loss.data
            
            output = output.data.cpu().numpy()
            for out in output:
                coords = extract_coords(out)
                s = coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence'])
                val_preds.append(s)
    
    valid_loss /= len(dev_loader.dataset)
    valid_mask_loss /= len(dev_loader.dataset)
    valid_regr_loss /= len(dev_loader.dataset)
    
    df_dev_ = df_dev.copy()
    df_dev_['PredictionString'] = val_preds
    filename = 'val_predictions_{}.csv'.format(epoch)
    df_dev_.to_csv(filename, index=False)
    
    map = calk_map(filename)
    
    writer.add_scalar('dev/loss', valid_loss, epoch)
    writer.add_scalar('dev/mask_loss', valid_mask_loss, epoch)
    writer.add_scalar('dev/regr_loss', valid_regr_loss, epoch)
    writer.add_scalar('dev/map', map, epoch)
    
    if history is not None:
        history.loc[epoch, 'mAP'] = map
        history.loc[epoch, 'dev_loss'] = valid_loss.cpu().numpy()
        history.loc[epoch, 'mask_loss'] = valid_mask_loss.cpu().numpy()
        history.loc[epoch, 'regr_loss'] = valid_regr_loss.cpu().numpy()
    
    print('Dev: mAP: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}'.format(
        map,
        valid_loss, 
        valid_mask_loss, 
        valid_regr_loss))
    
    return map





import gc

history = pd.DataFrame()

epoch = 0





def train_loop():
    global epoch
    map_max = -1
    epochs_since_last_improvement = 0
    best_cpname = None
    while True:
        torch.cuda.empty_cache()
        gc.collect()
        train(epoch, history)
        map = evaluate(epoch, history)
        cpname = os.path.join(CPFOLDER, "epoch_" + str(epoch) + ".pth")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, 
            cpname
        )
        epoch += 1
        if map > map_max:
            map_max = map
            epochs_since_last_improvement = 0
            best_cpname = cpname
        else:
            epochs_since_last_improvement += 1
            if epochs_since_last_improvement > patience:
                break
    return best_cpname





for p in model.base_model.parameters():
    p.requires_grad = False





best_cpname = train_loop()





print('Loading from', best_cpname)
sd = torch.load(best_cpname)
model.load_state_dict(sd['model'])
optimizer.load_state_dict(sd['optimizer'])





for p in model.base_model.layer4.parameters():
    p.requires_grad = True

for p in model.base_model.layer3.parameters():
    p.requires_grad = True





best_cpname = train_loop()





print('Loading from', best_cpname)
sd = torch.load(best_cpname)
model.load_state_dict(sd['model'])
optimizer.load_state_dict(sd['optimizer'])





sd = optimizer.state_dict()
sd['param_groups'][0]['lr'] = sd['param_groups'][0]['lr'] / 10
optimizer.load_state_dict(sd)





best_cpname = train_loop()





print('Loading from', best_cpname)
sd = torch.load(best_cpname)
model.load_state_dict(sd['model'])
optimizer.load_state_dict(sd['optimizer'])





sd = optimizer.state_dict()
sd['param_groups'][0]['lr'] = sd['param_groups'][0]['lr'] / 10
optimizer.load_state_dict(sd)





best_cpname = train_loop()





model.load_state_dict(torch.load(os.path.join(CPFOLDER, 'epoch_66.pth'))['model'])





predictions = []

test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=4)

model.eval()

for img, _, _ in tqdm(test_loader):
    with torch.no_grad():
        output = model(img.to(device))
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out)
        s = coords2str(coords, names=['yaw', 'pitch', 'roll', 'x', 'y', 'z', 'confidence'])
        predictions.append(s)





test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv('predictions.csv', index=False)
test.head()

