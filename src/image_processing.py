import gradio as gr
import pandas as pd
from pathlib import Path
import numpy as np
import cv2
from multiprocessing import Process, Queue
from groundingdino_crop import crop_imgs
from classify import predict
from outlier_detection import detect_outliers
from queue import Empty


N_STEPS = 3

def format_outlier_rows(row):
    if row['Outlier']:
        row['Genus'] = "Non Hymenoptera"
        row['Genus Score'] = ""
        row['Sex'] = ""
        row['Sex Score'] = ""
    return row

def process_images(img_paths: list[Path]):
    progress = gr.Progress()
    progress((0, N_STEPS), 'Starting the process...')

    queue = Queue()
    crop_process = Process(target=crop_imgs, args=(img_paths, queue))
    crop_process.start()
    try:
        queue.get(timeout=60)
    except Empty:
        raise Exception('Something went wrong loading the GroundingDINO model.')

    # load, crop and resize images
    imgs = []
    for img_path in progress.tqdm(img_paths, desc="Loading, cropping and resizing the images..."):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        crop = queue.get(timeout=60)
        x0, x1, y0, y1 = crop['x0'], crop['x1'], crop['y0'], crop['y1']
        img = img[y0:y1, x0:x1]
        img = cv2.resize(img, (224,224))
        imgs.append(img)

    imgs = np.array(imgs)

    progress((1, N_STEPS), 'Predicting the images...')

    tf_process = Process(target=predict, args=(imgs, queue))
    tf_process.start()
    tf_process.join()
    clf_df = queue.get(timeout=120)

    progress((2, N_STEPS), 'Running the outlier detection...')

    outlier_process = Process(target=detect_outliers, args=(imgs, queue))
    outlier_process.start()
    outlier_process.join()
    outlier_clf = queue.get(timeout=600)

    df = pd.DataFrame({
        'Image Path': [str(path) for path in img_paths],
    })

    df = pd.concat((df, clf_df), axis=1)
    df['Outlier'] = outlier_clf

    df = df.apply(format_outlier_rows, axis=1)

    return df