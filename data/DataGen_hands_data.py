import os
import sys
import json
import fnmatch
import tarfile
from PIL import Image
from glob import glob
from tqdm import tqdm
import cv2
from six.moves import urllib

import numpy as np

from utils import loadmat, imread, imwrite

DATA_FNAME = 'hands.npz'


def save_array_to_grayscale_image(array, path):
  Image.fromarray(array).convert('L').save(path)


def process_json_list(json_list, img):
  ldmks = [eval(s) for s in json_list]
  return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])


def maybe_preprocess(config, data_path, sample_path=None):
  if config.max_synthetic_num < 0:
    max_synthetic_num = None
  else:
    max_synthetic_num = config.max_synthetic_num
  
  # Real hands dataset
  real_imgs_dir = os.path.join(data_path, config.real_image_dir)
  synthetic_img_dir = os.path.join(data_path, config.synthetic_image_dir)
  synthetic_grayscale_path = synthetic_img_dir + '_grayscale'
  npz_path = os.path.join(data_path, DATA_FNAME)

  if not os.path.exists(npz_path):
    real_images_paths = os.listdir(real_imgs_dir)

    print("[*] Preprocessing real `DG hands` data...")
    real_images = []
    for img_path in real_images_paths:
        real_images.append(os.path.join(real_imgs_dir, img_path))

    print("\n[*] Finished preprocessing real `DG hands` data.")

    real_data = np.stack(real_images, axis=0)
    np.savez(npz_path, real=real_data)

  if not os.path.isdir(synthetic_img_dir+'_grayscale'):
    print("[*] Preprocessing synthetic `DG hands` data...")
    os.mkdir(synthetic_grayscale_path)
    for img in tqdm(os.listdir(synthetic_img_dir)):
      img_path = os.path.join(synthetic_img_dir, img)
      new_img_path = os.path.join(synthetic_grayscale_path, img.replace('.png', '_grayscale.png'))
      img_data = cv2.imread(img_path)
      img_data = cv2.resize(img_data, (config.input_width, config.input_height))
      save_array_to_grayscale_image(img_data, new_img_path)
    print("\n[*] Finished preprocessing synthetic `DG hands` data.")

  return synthetic_grayscale_path

def load(config, data_path, sample_path, rng):
  if not os.path.exists(data_path):
    print('creating folder', data_path)
    os.makedirs(data_path)

  synthetic_imgs_dir = maybe_preprocess(config, data_path, sample_path)
  real_data = np.load(os.path.join(data_path, DATA_FNAME))
  real_data = real_data['real']

  if not os.path.exists(sample_path):
    os.makedirs(sample_path)

  return real_data, synthetic_imgs_dir

class DataLoader(object):
  def __init__(self, config, rng=None):
    self.rng = np.random.RandomState(1) if rng is None else rng

    self.input_channel = config.input_channel
    self.input_height = config.input_height
    self.input_width = config.input_width

    self.data_path = os.path.join(config.data_dir, config.data_set)
    self.sample_path = os.path.join(self.data_path, config.sample_dir)
    self.batch_size = config.batch_size
    self.debug = config.debug

    self.real_data, synthetic_image_path = load(config, self.data_path, self.sample_path, rng)
    self.synthetic_data_paths = np.array(glob(os.path.join(synthetic_image_path, '*_grayscale.png')))
    self.synthetic_data_dims = list(imread(self.synthetic_data_paths[0]).shape) + [1]

    self.synthetic_data_paths.sort()

    if np.rank(self.real_data) == 3:
      self.real_data = np.expand_dims(self.real_data, -1)
    
    self.real_p = 0

  def get_observation_size(self):
    return self.real_data.shape[1:]

  def get_num_labels(self):
    return np.amax(self.labels) + 1

  def reset(self):
    self.real_p = 0

  def __iter__(self):
    return self

  def __next__(self, n=None):
    """ n is the number of examples to fetch """
    if n is None: n = self.batch_size

    if self.real_p == 0:
      inds = self.rng.permutation(self.real_data.shape[0])
      self.real_data = self.real_data[inds]

    if self.real_p + n > self.real_data.shape[0]:
      self.reset()

    x = self.real_data[self.real_p : self.real_p + n]
    self.real_p += self.batch_size
    output_imgs = []
    for img_path in x:
      img_data = cv2.imread(img_path)
      if self.input_channel == 1:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
      else:
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
      img_data = cv2.resize(img_data, (self.input_width, self.input_height))  # resize and normalize values
      output_imgs.append(img_data)
    return output_imgs

  next = __next__
