'''
  Created by irving on 8/10/18
'''
import os
import pickle

num_of_image = 3
count = 0
dataset_dir = '/mnt/sda/dataset/openimage_body_face'

with open('body_face_sample.pickle', 'rb') as f:
    body_face_samples = pickle.load(f)

for key, value in body_face_samples.items():
    # os.system('wget ' + value.remote_path + ' -P ' + dataset_dir)
    value.load()
    count += 1
    if count > num_of_image:
        break
