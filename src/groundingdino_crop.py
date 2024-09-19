import numpy as np
from pathlib import Path
from multiprocessing import Queue
import sys

from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert


BASE_PATH = Path(__file__).parents[1] / 'weights'
CONFIG_PATH = Path(sys.prefix) / 'Lib/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py'
WEIGHTS_PATH = BASE_PATH / 'groundingdino_swint_ogc.pth'

TEXT_PROMPT = "insect. wasp. wings."
BOX_TRESHOLD = 0.29
TEXT_TRESHOLD = 0.25


def get_all_including_box(boxes: np.ndarray):
    x0, y0, x1, y1 = boxes[0]
    for box in boxes:
        x0 = box[0] if x0 > box[0] else x0
        y0 = box[1] if y0 > box[1] else y0
        x1 = box[2] if x1 < box[2] else x1
        y1 = box[3] if y1 < box[3] else y1

    return x0, y0, x1, y1

def crop_imgs(img_paths: list[Path], queue: Queue):
    model = load_model(CONFIG_PATH, WEIGHTS_PATH, device='cpu')

    queue.put('done')

    for path in img_paths:
        image_source, image = load_image(path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device='cpu'
        )

        h, w, _ = image_source.shape
        boundingboxes = boxes * np.array([w, h, w, h])
        converted_boxes = box_convert(boxes=boundingboxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)
        try:
            x0, y0, x1, y1 = get_all_including_box(converted_boxes)
        except IndexError:
            x0, y0, x1, y1 = 0, 0, w, h
            
        margin = 25
        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, w)
        y1 = min(y1 + margin, h)

        crop = {
            'x0': int(x0),
            'x1': int(x1),
            'y0': int(y0),
            'y1': int(y1),
        }

        queue.put(crop)