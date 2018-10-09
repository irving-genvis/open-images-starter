#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run some basic statistical analysis on the samples.
"""

import os
from typing import Dict

from modules.loader import Loader
from modules.settings import ProjectSettings
from tools.util.logger import Logger
import matplotlib.pyplot as plt
import numpy as np
import pickle

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"

# body_labels = ['MAN', 'WOMAN', 'PERSON', 'GIRL', 'BOY']
body_labels = ['/m/04yx4', '/m/03bt1vf', '/m/01g317', '/m/05r655', '/m/01bl7v']

# face_labels = ['HUMAN FACE', 'HUMAN HEAD']
face_labels = ['/m/0dzct', '/m/04hgtk']

# car_labels = ['Land vehicle']
car_labels = ['/m/01prls']

if __name__ == "__main__":

    # Load the project settings and required modules.
    Logger.log_special("Running Sample Analysis", with_gap=True)
    settings = ProjectSettings("settings.yaml")

    # Load the class labels.
    loader = Loader()
    loader.load_labels(settings.LABELS_FILE)

    # Get ALL of the samples in the directory.
    samples = []
    sample_files = os.listdir(settings.SAMPLES_DIRECTORY)
    for i in sample_files[:20]:
        file_path = os.path.join(settings.SAMPLES_DIRECTORY, i)
        samples += Loader.load_sample_set_from_file(file_path)

    class_instances = {}
    class_appearances = {}

    for key in loader.label_map:
        class_instances[key] = 0
        class_appearances[key] = 0

    num_body_samples = 0
    num_face_samples = 0
    num_car_samples = 0
    num_body_face_samples = 0

    body_face_sample = {}
    car_sample = {}

    for sample in samples:
        body_flag = False
        face_flag = False
        car_flag = False

        for region in sample.detect_regions:
            if region.class_id in body_labels:
                body_flag = True
            if region.class_id in face_labels:
                face_flag = True
            if region.class_id in car_labels:
                car_flag = True

        if body_flag:
            num_body_samples += 1
        if face_flag:
            num_face_samples += 1
        if car_flag:
            num_car_samples += 1
            car_sample[sample.key] = sample
        if body_flag and face_flag:
            num_body_face_samples += 1
            body_face_sample[sample.key] = sample

    with open('body_face_sample.pickle', 'wb') as f:
        pickle.dump(body_face_sample, f)
    with open('car_sample.pickle', 'wb') as f:
        pickle.dump(car_sample, f)

    print(len(body_face_sample))
    print(len(car_sample))

    custom_samples = body_face_sample.copy()
    custom_samples.update(car_sample)
    print(len(body_face_sample))
    print(len(car_sample))
    print(len(custom_samples))


    print('body: ', num_body_samples)
    print('face: ', num_face_samples)
    print('car: ', num_car_samples)
    print('b+f: ', num_body_face_samples)
