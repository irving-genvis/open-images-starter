'''

  Created by irving on 8/10/18

'''
import pickle
import cv2
from modules.sample import Sample
from modules.loader import Loader
from modules.settings import ProjectSettings

dataset_dir = '/mnt/sda/dataset/openimage_body_face'

with open('body_face_sample.pickle', 'rb') as f:
    body_face_samples: {str: Sample} = pickle.load(f)

    settings = ProjectSettings("settings.yaml")

    # Load the label mapping.
    loader = Loader()
    loader.load_labels(settings.LABELS_FILE)

body_face_labels = ['/m/04yx4', '/m/03bt1vf', '/m/01g317', '/m/05r655', '/m/01bl7v',
                    '/m/0dzct', '/m/04hgtk']

for key, value in body_face_samples.items():
    cv2.imshow('Vis', value.get_visualized_image_custom_label(label_map_function=loader.get_label,
                                                              custom_label=body_face_labels))
    cv2.waitKey(0)
