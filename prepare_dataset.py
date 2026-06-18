import os
import json
import random
import shutil
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm
from class_mapping import CLASS_MAPPING


BASE_DIR = "./data"
OUTPUT_DIR = "./yolo_dataset"

TRAIN_RATIO = 0.85

# коэффициенты прореживания
SAMPLE_RATES = {
    "part_1": 3.0,  # размеченные данные (x3)
    "part_2": 1.0,  # очищенный вручную набор №2
    "part_3": 1.0,  # мал. набор с холодильниками №3
    "part_4": 0.2,  # много ингред. на чистом фоне (20%)
    "part_5": 0.7,  # набор с холодильниками, но с хорошим освещением (70%)
    "part_6": 1.0,  # набор с холодильниками №6
    "part_7": 1.0,  # небольшой набор с реальными холодильниками №7
    "part_8": 1.0,  # набор с продуктами в тележках
}

# функции для приведения к единому формату
def get_unique_classes(mapping):
    return sorted(list(set(mapping.values())))


def coco_to_yolo_format(x_min, y_min, width, height, img_w, img_h):
    x_center = (x_min + width / 2) / img_w
    y_center = (y_min + height / 2) / img_h
    w_norm = width / img_w
    h_norm = height / img_h
    return x_center, y_center, w_norm, h_norm


def process_coco_part(part_name, json_paths, img_dirs, sample_rate):
    """Обрабатывает части с COCO JSON форматом"""
    collected_data = []

    for json_path, img_dir in zip(json_paths, img_dirs):
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)

        # id категории -> имя категории -> единое имя для всех данных
        cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        img_id_to_info = {img['id']: img for img in coco_data['images']}

        # группировка по изображениям
        anns_by_img = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in anns_by_img:
                anns_by_img[img_id] = []
            anns_by_img[img_id].append(ann)

        valid_img_ids = list(anns_by_img.keys())
        if sample_rate < 1.0:
            valid_img_ids = random.sample(valid_img_ids, int(len(valid_img_ids) * sample_rate))

        for img_id in tqdm(valid_img_ids, desc=f"Processing {part_name} (COCO)"):
            img_info = img_id_to_info[img_id]
            img_filename = img_info['file_name']
            img_path = os.path.join(img_dir, img_filename)

            if not os.path.exists(img_path):
                # иногда file_name содержит подпапки, пробуем найти просто по имени
                img_path = os.path.join(img_dir, os.path.basename(img_filename))
                if not os.path.exists(img_path):
                    continue

            img_w, img_h = img_info['width'], img_info['height']
            yolo_annotations = []

            for ann in anns_by_img[img_id]:
                cat_name = cat_id_to_name.get(ann['category_id'], "unknown")
                unified_name = CLASS_MAPPING.get(cat_name)

                if unified_name:
                    x_min, y_min, w, h = ann['bbox']
                    xc, yc, wn, hn = coco_to_yolo_format(x_min, y_min, w, h, img_w, img_h)
                    yolo_annotations.append((unified_name, xc, yc, wn, hn))

            if yolo_annotations:  # только если есть валидные аннотации
                collected_data.append((img_path, yolo_annotations, part_name))

    return collected_data


def process_voc_part(part_name, xml_dir, img_dir, sample_rate):
    """Обрабатывает части с Pascal VOC XML форматом"""
    collected_data = []
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]

    if sample_rate < 1.0:
        xml_files = random.sample(xml_files, int(len(xml_files) * sample_rate))

    for xml_file in tqdm(xml_files, desc=f"Processing {part_name} (VOC)"):
        xml_path = os.path.join(xml_dir, xml_file)
        img_filename = xml_file.replace('.xml', '.jpg')

        img_path = os.path.join(img_dir, img_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, xml_file.replace('.xml', '.png'))
            if not os.path.exists(img_path):
                continue

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # получаем реальные размеры изображения
        img_cv = cv2.imread(img_path)
        if img_cv is None:
            continue
        img_h, img_w = img_cv.shape[:2]

        yolo_annotations = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            unified_name = CLASS_MAPPING.get(name)

            if unified_name:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                w = xmax - xmin
                h = ymax - ymin
                xc, yc, wn, hn = coco_to_yolo_format(xmin, ymin, w, h, img_w, img_h)
                yolo_annotations.append((unified_name, xc, yc, wn, hn))

        if yolo_annotations:
            collected_data.append((img_path, yolo_annotations, part_name))

    return collected_data


# +++ ПРОЦЕСС СБОРА +++
print("Сбор и конвертация датасета...")
all_data = []

# part_1: сливаем default и Train
p1_jsons = [
    os.path.join(BASE_DIR, "part_1/annotations/instances_default.json"),
    os.path.join(BASE_DIR, "part_1/annotations/instances_Train.json")
]
p1_imgs = [
    os.path.join(BASE_DIR, "part_1/images/default"),
    os.path.join(BASE_DIR, "part_1/images/Train")
]
all_data.extend(process_coco_part("part_1", p1_jsons, p1_imgs, SAMPLE_RATES["part_1"]))

# part_2
all_data.extend(process_voc_part(
    "part_2",
    os.path.join(BASE_DIR, "part_2/annotations"),
    os.path.join(BASE_DIR, "part_2/images"),
    SAMPLE_RATES["part_2"]
))

# part_3
p3_jsons = [
    os.path.join(BASE_DIR, "part_3/test/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_3/valid/_annotations.coco.json")
]
p3_imgs = [
    os.path.join(BASE_DIR, "part_3/test"),
    os.path.join(BASE_DIR, "part_3/valid")
]
all_data.extend(process_coco_part("part_3", p3_jsons, p3_imgs, SAMPLE_RATES["part_3"]))

# part_4
p4_jsons = [
    os.path.join(BASE_DIR, "part_4/train/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_4/test/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_4/valid/_annotations.coco.json")
]
p4_imgs = [
    os.path.join(BASE_DIR, "part_4/train"),
    os.path.join(BASE_DIR, "part_4/test"),
    os.path.join(BASE_DIR, "part_4/valid")
]
all_data.extend(process_coco_part("part_4", p4_jsons, p4_imgs, SAMPLE_RATES["part_4"]))

# part_5
p5_jsons = [
    os.path.join(BASE_DIR, "part_5/valid/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_5/test/_annotations.coco.json")
]
p5_imgs = [
    os.path.join(BASE_DIR, "part_5/valid"),
    os.path.join(BASE_DIR, "part_5/test")
]
all_data.extend(process_coco_part("part_5", p5_jsons, p5_imgs, SAMPLE_RATES["part_5"]))

# part_6
p6_jsons = [
    os.path.join(BASE_DIR, "part_6/train/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_6/valid/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_6/test/_annotations.coco.json")
]
p6_imgs = [
    os.path.join(BASE_DIR, "part_6/train"),
    os.path.join(BASE_DIR, "part_6/valid"),
    os.path.join(BASE_DIR, "part_6/test")
]
all_data.extend(process_coco_part("part_6", p6_jsons, p6_imgs, SAMPLE_RATES["part_6"]))

# part_7
p7_jsons = [
    os.path.join(BASE_DIR, "part_7/train/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_7/valid/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_7/test/_annotations.coco.json")
]
p7_imgs = [
    os.path.join(BASE_DIR, "part_7/train"),
    os.path.join(BASE_DIR, "part_7/valid"),
    os.path.join(BASE_DIR, "part_7/test")
]
all_data.extend(process_coco_part("part_7", p7_jsons, p7_imgs, SAMPLE_RATES["part_7"]))

# part_8
p8_jsons = [
    os.path.join(BASE_DIR, "part_8/train/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_8/valid/_annotations.coco.json"),
    os.path.join(BASE_DIR, "part_8/test/_annotations.coco.json")
]
p8_imgs = [
    os.path.join(BASE_DIR, "part_8/train"),
    os.path.join(BASE_DIR, "part_8/valid"),
    os.path.join(BASE_DIR, "part_8/test")
]
all_data.extend(process_coco_part("part_8", p8_jsons, p8_imgs, SAMPLE_RATES["part_8"]))


# разделение на выборки
print(f"\nВсего собрано валидных изображений: {len(all_data)}")

random.seed(42)
random.shuffle(all_data)
split_idx = int(len(all_data) * TRAIN_RATIO)
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

print(f"Train: {len(train_data)} изображений")
print(f"Val: {len(val_data)} изображений")


# сохранение в формате yolo
unique_classes = get_unique_classes(CLASS_MAPPING)
class_to_id = {cls: idx for idx, cls in enumerate(unique_classes)}
print(f"\nИтоговые классы ({len(unique_classes)} шт.): {unique_classes}")

for split, data in [("train", train_data), ("val", val_data)]:
    img_out_dir = os.path.join(OUTPUT_DIR, "images", split)
    lbl_out_dir = os.path.join(OUTPUT_DIR, "labels", split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    # поскольку рамеченных реальных данных немного, то возьмем их в обучающей выборке 3 раза
    if split == "train":
        expanded_data = []
        for img_path, annotations, part_name in data:
            sample_rate = SAMPLE_RATES.get(part_name, 1.0)

            if sample_rate > 1.0:
                n_copies = int(sample_rate)
                for _ in range(n_copies):
                    expanded_data.append((img_path, annotations, part_name))
            else:
                expanded_data.append((img_path, annotations, part_name))

        data = expanded_data
        print(f"После oversampling в train: {len(data)} фото")

    for img_path, annotations, part_name in tqdm(data, desc=f"Saving {split} set"):
        # копируем изображение
        img_filename = os.path.basename(img_path)
        # делаем имя уникальным
        unique_img_name = f"{split}_{random.randint(10000, 99999)}_{img_filename}"
        dst_img_path = os.path.join(img_out_dir, unique_img_name)
        shutil.copy2(img_path, dst_img_path)

        # создаем .txt файл
        txt_filename = os.path.splitext(unique_img_name)[0] + ".txt"
        dst_txt_path = os.path.join(lbl_out_dir, txt_filename)

        with open(dst_txt_path, 'w') as f:
            for cls_name, xc, yc, w, h in annotations:
                cls_id = class_to_id[cls_name]
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

print("\nНабор данных успешно подготовлен для YOLOv11")
print(f"Путь к директории: {os.path.abspath(OUTPUT_DIR)}")