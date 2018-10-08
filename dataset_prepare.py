'''

  Created by irving on 8/10/18

'''
import argparse
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
import os
import pickle
from modules.sample import Sample

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--dataset_folder", type=str, help="dataset folder")
    return parser.parse_args()


args = get_args()

def write_to_xml(folder_path, fn, bboxes, names, img_size=[1920, 1080, 3], dataset='NA'):
    '''
    Input an image and all the annotations, convert it to xml
    :param fn: filename *.jpg *.png ...
    :param bbox: [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax] ...]
    :param nm: ['person', 'person' ...]
    :param img_size: [width, height, depth]
    :return: None
    '''
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder')
    folder.text = dataset

    filename = ET.SubElement(annotation, 'filename')
    filename.text = fn

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = dataset
    source_annotation = ET.SubElement(source, 'source_annotation')
    source_annotation.text = dataset
    image = ET.SubElement(source, 'image')
    image.text = dataset

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(img_size[0])
    height = ET.SubElement(size, 'height')
    height.text = str(img_size[1])
    depth = ET.SubElement(size, 'depth')
    depth.text = str(img_size[2])
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        nm = names[i]
        object = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object, 'name')
        name.text = nm
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Frontal'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = '0'
        occluded = ET.SubElement(object, 'occluded')
        occluded.text = '0'
        bndbox = ET.SubElement(object, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(bbox[0])
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(bbox[1])
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(bbox[2])
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(bbox[3])
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = '0'

    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(folder_path, (fn.split('.')[0] + '.xml')))


def convert_sample_to_xml(s: Sample, dataset_dir):
    # body_labels = ['MAN', 'WOMAN', 'PERSON', 'GIRL', 'BOY']
    body_labels = ['/m/04yx4', '/m/03bt1vf', '/m/01g317', '/m/05r655', '/m/01bl7v']

    # face_labels = ['HUMAN FACE', 'HUMAN HEAD']
    face_labels = ['/m/0dzct', '/m/04hgtk']

    # car_labels = ['Land vehicle']
    car_labels = ['/m/01prls']

    img = s.image
    height, width, channel = img.shape
    xml_path = os.path.join(dataset_dir, 'annotations', 'xmls')

    bbox_list = []
    cls_list = []

    for region in sample.detect_regions:
        if region.is_group_of:
            continue
        elif region.class_id in body_labels:
            cls_list.append('person')
        elif region.class_id in face_labels:
            cls_list.append('face')
        elif region.class_id in car_labels:
            cls_list.append('car')
        else:
            continue

        bbox_list.append([int(float(region.left * width)),
                          int(float(region.top * height)),
                          int(float(region.right * width)),
                          int(float(region.bottom * height))])

    write_to_xml(xml_path, s.key + '.jpg', bbox_list, cls_list,
                 img_size=[width, height, channel])


dataset_dir = args.dataset_folder

with open('body_face_sample.pickle', 'rb') as f:
    body_face_samples: {str: Sample} = pickle.load(f)

with open('car_sample.pickle', 'rb') as f:
    car_samples: {str: Sample} = pickle.load(f)

custom_samples = body_face_samples.copy()
custom_samples.update(car_samples)

if not os.path.exists(dataset_dir):
    os.mkdir(dataset_dir)
if not os.path.exists(os.path.join(dataset_dir, 'annotations')):
    os.mkdir(os.path.join(dataset_dir, 'annotations'))
if not os.path.exists(os.path.join(dataset_dir, 'annotations', 'trimaps')):
    os.mkdir(os.path.join(dataset_dir, 'annotations', 'trimaps'))
if not os.path.exists(os.path.join(dataset_dir, 'annotations', 'xmls')):
    os.mkdir(os.path.join(dataset_dir, 'annotations', 'xmls'))

with open(os.path.join(dataset_dir, 'annotations', 'trainval.txt'), 'w') as f:
    for key, sample in custom_samples.items():
        # convert sample to xml annotations
        f.write(key + '.jpg\n')
        convert_sample_to_xml(sample, dataset_dir)
        os.system('mv ' + sample._local_path + ' ' + dataset_dir + '/images')
        # download images
        # os.system('wget ' + sample.remote_path + ' -P ' + dataset_dir + '/images')

