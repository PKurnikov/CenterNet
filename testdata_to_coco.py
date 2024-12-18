import numpy as np
import cv2
import json
import copy
import datetime

import random

import os
import numpy as np
import cv2
import shutil

import xml.etree.ElementTree as ET

from shapely.geometry import Polygon

def is_dig(n):
  try:
    int(n)
    return True 
  except ValueError:
    return False

def find_files(catalog, f):
    find_files = []
    for root, dirs, files in os.walk(catalog):
        find_files += [os.path.join(root, name) for name in files if (f in name)]
    return find_files


def bounding_box(points):
    x_coordinates = []
    y_coordinates = []
    for i in range(len(points[0]) // 2):
      x = points[0][i * 2]
      y = points[0][i * 2 + 1]
      x_coordinates.append(x)
      y_coordinates.append(y)
    #x_coordinates, y_coordinates = zip(*points)

    return [(min(x_coordinates), min(y_coordinates)), (max(x_coordinates), max(y_coordinates))]

def convert_datetime_to_string(dt=datetime.datetime.now(), formt="%Y-%m-%d %H:%M:%S"):
    return dt.strftime(formt)

def get_filtered_img_id(coco_annot, classes, filtered):
  images_ids = [None] * 1000000
  fin = []

  for data in coco_annot.data['annotations']:
    category_id = data['category_id']
    category_id = get_cat_id(category_id, coco_annot.data['categories'])

    image_id = data['image_id']
    annot_id = data['id']

    try:
      image_cls = coco_class_list[category_id]
      #if (image_cls in filtered) or (image_cls in ['car', 'bus', 'truck']):
      try:
        if (images_ids[image_id] is None):
          images_ids[image_id] = []
        images_ids[image_id].append(image_cls)
      except:
        print("except")
      #else:
      #  try:
      #    images_ids.remove(image_id)
      #  except:
      #    print('value not in list')
    except:
      print("except")

  for ids, images_id in enumerate(images_ids):
    try:
      #if (('motorcycle' in images_id) and ):
      #  continue
      hist = {}
      for cls in images_id:
        try:
          hist[cls] += 1
        except:
          hist[cls] = 0
          hist[cls] += 1

      if (('airplane' in images_id)):
        continue
      if (('boat' in images_id)):
        continue
      if (('train' in images_id)):
        continue
      if (('train' in images_id)):
        continue
      if (('train' in images_id)):
        continue
      if (('train' in images_id)):
        continue
      if (('horse' in images_id)):
        continue
      if (('sheep' in images_id)):
        continue
      if (('cow' in images_id)):
        continue
      if (('elephant' in images_id)):
        continue
      if (('bear' in images_id)):
        continue
      if (('zebra' in images_id)):
        continue
      if (('bed' in images_id)):
        continue

      if ((('person' in images_id) and ('car' in images_id)) or 
        (('person' in images_id) and ('bus' in images_id)) or 
        (('person' in images_id) and ('truck' in images_id)) or
        (('person' in images_id) and ('motorcycle' in images_id)) or
        (('person' in images_id) and ('train' in images_id)) or
        (('person' in images_id) and ('bicycle' in images_id))):
        
        person_count = 0
        machinerry_count = 0
        byke_count = 0
        for n, v in hist.items():
          if n in ["person"]:
            person_count = hist["person"]
          if n in ["bus"]:
            machinerry_count += hist["bus"]
          if n in ["truck"]:
            machinerry_count += hist["truck"]
          if n in ["train"]:
            machinerry_count += hist["train"]
          if n in ["motorcycle"]:
            byke_count += hist["motorcycle"]
          if n in ["bicycle"]:
            byke_count += hist["bicycle"]

        #if (ids == 165681):
        #  print()

        if (byke_count > 5):
          continue
        if (person_count < 8):
          continue
        if (person_count / (machinerry_count + 0.000001) < 1.0):
          continue
        if (person_count / (byke_count + 0.000001) < 5.0):
          continue
        fin.append(ids)
    except:
      test = None

  return fin

def initialize_coco_annotation(config):
    """Инициализация объекта аннотаций COCO."""
    coco_annot = CocoAnnotationClass(config["coco_class_list_reduced"], "coco")
    max_img_id = -1
    max_annot_id = -1
    for v in coco_annot.data['images']:
      if v['id'] > IMG_ID:
        IMG_ID = v['id']

    for v in coco_annot.data['annotations']:
      if v['id'] > ANNOT_ID:
        ANNOT_ID = v['id']

    return coco_annot, max_img_id + 1, max_annot_id + 1

def find_frame_objects(tree, file_path):
    """
    Извлекает объекты аннотации из файла разметки для указанного кадра.

    :param tree: Объект дерева XML (ET.parse).
    :param file_path: Путь к текущему изображению (для извлечения номера кадра).
    :return: Список объектов из разметки для текущего кадра.
    """
    try:
        # Определение номера кадра из имени файла
        frame_number = int(os.path.basename(file_path).split('.')[-2])

        # Поиск объектов для конкретного кадра
        frame_data_array = tree._root.findall('FrameDataArray')
        if not frame_data_array:
            return []

        for frame in frame_data_array[0]:
            frame_number_ = int(frame.find('FrameNumber').text)
            if frame_number_ == frame_number:
                return frame.find('FrameObjects')

        # Если объекты для указанного кадра не найдены
        return []
    except Exception as e:
        print(f"Error in find_frame_objects: {e}")
        return []

def load_markup_file(data_dir, file_path):
    """Загрузка и парсинг файла разметки."""
    try:
        mar_folder_name = os.path.basename(file_path)[:-11] + '.avi.dat'
        mar_name = os.path.basename(file_path)[:-11] + '.avi.markup.xml'
        part = file_path.split('\\')[-3]
        mar_path = os.path.join(data_dir, part, mar_folder_name, mar_name)
        return ET.parse(mar_path)
    except Exception as e:
        print(f"Failed to load markup: {e}")
        return None

def create_polygon(vertices_, src_shape):
    """Создание полигонов с ограничением по границам изображения."""
    polygon = Polygon([(x, y) for x, y in zip(vertices_[::2], vertices_[1::2])])
    if not polygon.is_valid:
        polygon = Polygon(polygon.convex_hull)

    intersection = polygon.intersection(Polygon([(0, 0), (src_shape[1], 0), (src_shape[1], src_shape[0]), (0, src_shape[0]), (0, 0)]))
    rect = intersection.bounds
    l_top_x, r_bottom_x = min(rect[0], rect[2]), max(rect[0], rect[2])
    l_top_y, r_bottom_y = min(rect[1], rect[3]), max(rect[1], rect[3])
    polygon = np.array([[l_top_x, l_top_y], [r_bottom_x, l_top_y], [r_bottom_x, r_bottom_y], [l_top_x, r_bottom_y]], dtype=np.int32)
    polygon[polygon < 0] = 0
    return polygon

def process_frame_objects(frame_objects, frame_number, src_shape):
    """Обработка объектов в кадре."""
    kps = CONFIG["keypoints_template"]
    vertices_ = []
    num_keypoints = 0
    for frame_obj in frame_objects:
        try:
            rect_str = frame_obj.findall('rect')[0].text
            vertices = [int(word) if is_dig(word) else 0 for word in rect_str.split()]
            x, y, w, h = vertices
        except:
            vertices = [0, 0, 0, 0]

        try:
            subType = frame_obj.findall('subType')[0].text
            properties = frame_obj.findall('properties')[0].text if frame_obj.findall('properties') else "normal"
            is_visible = {"truncated": 0, "occluded": 1, "normal": 2}.get(properties, 0)
            if is_visible:
                num_keypoints += 1

            idx_map = {"left_up": 0, "left_bottom": 1, "center": 2, "right_bottom": 3, "right_up": 4}
            if subType in idx_map:
                idx = idx_map[subType]
                kps[idx] = [x, y, is_visible]
                vertices_.extend([x, y])
        except:
            continue

    # Создание полигонов и ограничений
    polygon = create_polygon(vertices_, src_shape)
    return kps, polygon, num_keypoints

class CocoAnnotationClass(object):
    def __init__(self, classes, supercategory=""):
        self.classes = classes
        self.data = self._get_default_data()
        self._init_var()
        for c,idx in self.map_classes_idx.items():
            self._add_category(c,idx,supercategory)

    def _init_var(self):
        self.map_classes_idx = {c: ix+1 for ix,c in enumerate(self.classes)}  # coco is 1-indexed
        self.map_idx_classes = {v:k for k,v in self.map_classes_idx.items()}

    def _get_default_data(self):
        default_d = {
            "info": {
                "year" : 2019, 
                "version" : "", 
                "description" : "", 
                "contributor" : "", 
                "url" : "", 
                "date_created" : convert_datetime_to_string()
            },
            "images": [],
            "annotations": [],
            "categories": [],
            "licenses": [
                {
                    "id" : 1, 
                    "name" : "", 
                    "url" : ""
                }
            ]
        }
        return default_d

    def set_classes(self, classes):
        self.classes = classes

    def clear(self):
        self.data = self._get_default_data()

    def _add_category(self, name, id=None, supercategory=""):
        cat_id = len(self.data["categories"]) + 1 if id is None else id
        cat_data = {
                    "id" : cat_id, 
                    "name" : name, 
                    "supercategory" : supercategory
                }
        self.data["categories"].append(cat_data)

    def add_annot(self, id, img_id, img_cls, seg_data, kps, meta_data={}, is_crowd=0):
        """
        CAN NOW SUPPORT MULTIPLE SEG POLYGONS
        DEPRECATED: ONLY SUPPORTS seg polygons of len 1 i.e. cannot support multiple polygons that refer to the same id"""
        if isinstance(img_cls, str):
            if img_cls not in self.map_classes_idx:
                print("%s not in coco classes!"%(img_cls))
                return 
            cat_id = self.map_classes_idx[img_cls]
        else:
            assert img_cls in self.map_idx_classes
            cat_id = img_cls
        # seg_data_arr = np.array(seg_data)
        # if len(seg_data_arr.shape) == 2:
        #     seg_data_arr = seg_data_arr[None,:]
        # assert seg_data_arr.shape[-1] == 2 # x,y
        if len(seg_data) == 0:
            print("Polygon (seg_data) is empty!")
            return 
        seg_data_arr = seg_data if type(seg_data[0][0]) in [list, np.ndarray] else [seg_data]
        concat_arr = np.concatenate(seg_data_arr)
        bbox = np.array([np.amin(concat_arr, axis=0), np.amax(concat_arr, axis=0)]).reshape(4)
        bbox[2:] -= bbox[:2]
        bbox = bbox.tolist()
        area = sum([cv2.contourArea(arr) for arr in seg_data_arr])
        keypoints = [coord for x, y, _ in kps for coord in (x, y, _)]
        annot_data =    {
                    "id" : id,
                    "image_id" : img_id,
                    "category_id" : cat_id,
                    "segmentation" : [arr.flatten().tolist() for arr in seg_data_arr],
                    "area" : area,
                    "bbox" : bbox,
                    "iscrowd" : is_crowd,
                    "keypoints" : keypoints,
                    "meta" : meta_data
                }
        self.data["annotations"].append(annot_data)

    def add_image(self, id, width, height, file_name, meta_data={}, date_captured=convert_datetime_to_string()):
        img_data =  {
                    "id" : id,
                    "width" : width,
                    "height" : height,
                    "file_name" : file_name,
                    "license" : 1,
                    "flickr_url" : "",
                    "coco_url" : "",
                    "date_captured" : date_captured,
                    "meta": meta_data
                }

        self.data["images"].append(img_data)

    def get_annot_json(self):
        return copy.deepcopy(self.data)

    def save(self, out_file):
        with open(out_file, "w") as f:
            json.dump(self.data, f)
            print("Saved to %s"%(out_file))

    def load(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)
            print("Loaded from %s"%(json_file))
        self.classes = [c['name'] for c in self.data["categories"]]
        self._init_var()

coco_class_list = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane',
  'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
  'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
  'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
  'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
  'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
  'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
  'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
  'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
  'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Конфигурация
CONFIG = {
    "coco_class_list_reduced": ['broom'],
    "data_dir": r'W:/box/kurnikov/trm/SNOW/keypoints/data/',
    "mar_dir": r'W:/box/kurnikov/trm/SNOW/keypoints/data/ge2-pnt-679',
    "images_dir": 'IMAGES',
    "split_train_prob": 0.9,
    "default_class": "broom",
    "mean": [128.0, 128.0, 128.0],
    "scale": 0.0039,
    "keypoints_template": [[0, 0, 0]] * 5
}

def main():
    # Инициализация
    coco_annot, img_id, annot_id = initialize_coco_annotation(CONFIG)
    files = find_files(CONFIG["mar_dir"], 'png') + find_files(CONFIG["mar_dir"], 'jpg')
    total_files = len(files)
    assotiate_list = []

    for idx, file_path in enumerate(files[int(total_files * CONFIG["split_train_prob"]):]):
        print(f"Processing {int((idx / total_files) * 100)}%")
        src = cv2.imread(file_path)
        height, width = src.shape[:-1]

        # Загрузка файла разметки
        markup_tree = load_markup_file(CONFIG["data_dir"], file_path)
        if markup_tree:
            frame_objects = find_frame_objects(markup_tree, file_path)
        else:
            frame_objects = []

        # Обработка аннотаций
        if frame_objects:
            kps, polygon, num_keypoints = process_frame_objects(frame_objects, int(os.path.basename(file_path).split('.')[-2]), src.shape)
            coco_annot.add_annot(annot_id, img_id, CONFIG["default_class"], polygon, kps)
            annot_id += 1

        # Сохранение изображения
        file_name = f'{img_id:012d}.jpg'
        coco_annot.add_image(img_id, width, height, file_name)
        cv2.imwrite(os.path.join(CONFIG["data_dir"], CONFIG["images_dir"], file_name), src)
        img_id += 1

        assotiate_list.append(f"{os.path.basename(file_path)}\t{file_name}")

    # Сохранение аннотаций
    coco_annot.save(os.path.join(CONFIG["data_dir"], 'keypoints.json'))

if __name__ == "__main__":
    main()