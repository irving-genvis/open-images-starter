'''

  Created by irving on 8/10/18

'''
import pickle
import cv2
from modules.sample import Sample
from modules.loader import Loader
from modules.settings import ProjectSettings
import random

with open('body_face_sample.pickle', 'rb') as f:
    body_face_samples: {str: Sample} = pickle.load(f)
with open('car_sample.pickle', 'rb') as f:
    car_samples: {str: Sample} = pickle.load(f)

custom_samples = body_face_samples.copy()
custom_samples.update(car_samples)

keys = list(custom_samples.keys())
random.shuffle(keys)
custom_samples = {key: custom_samples[key] for key in keys}

settings = ProjectSettings("settings.yaml")

# Load the label mapping.
loader = Loader()
loader.load_labels(settings.LABELS_FILE)

body_face_labels = ['/m/04yx4', '/m/03bt1vf', '/m/01g317', '/m/05r655', '/m/01bl7v',
                    '/m/0dzct', '/m/04hgtk']

car_labels = ['/m/01prls']

for key, value in custom_samples.items():
    labelled_image = value.get_visualized_image_custom_label(label_map_function=loader.get_label,
                                                             custom_label=car_labels + body_face_labels)
    cv2.imwrite(ProjectSettings.instance().CUSTOM_LABELLED_DIRECTORY +
                key + '.jpg', labelled_image)
    cv2.imshow('Vis', labelled_image)
    cv2.waitKey(0)
