'''

  Created by irving on 9/10/18

'''
import os
import pickle
from modules.sample import Sample
from multiprocessing import Pool
from modules.settings import ProjectSettings

with open('body_face_sample.pickle', 'rb') as f:
    body_face_samples: {str: Sample} = pickle.load(f)

with open('car_sample.pickle', 'rb') as f:
    car_samples: {str: Sample} = pickle.load(f)

custom_samples = body_face_samples.copy()
custom_samples.update(car_samples)

download_path = [sample.remote_path for key, sample in custom_samples.items()]


# print(download_path)
# print(len(download_path))

def download(remote_path):
    print('wget ' + remote_path + ' -P ' + ProjectSettings.instance().CUSTOM_STORAGE_DIRECTORY)
    os.system('wget ' + remote_path + ' -P ' + ProjectSettings.instance().CUSTOM_STORAGE_DIRECTORY)

pool = Pool(os.cpu_count())
pool.map(download, download_path)