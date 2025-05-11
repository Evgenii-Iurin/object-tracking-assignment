from fastapi import FastAPI, WebSocket
from hungarian_tracker import Tracker
from tracker_strong import TrackerStrong
from metric import metric_fn
from experiments.tracks_t20_r5_s50 import track_data, country_balls_amount
import asyncio
import glob

import random
from PIL import Image
from io import BytesIO
import re
import base64
import json
import os
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')


def tracker_soft(el, tracker: Tracker):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    detections = el['data']
    frame_id = el['frame_id']
    el['data'] = tracker.process_frame(detections, frame_id)
    
    return el


def process_tracker_strong(el, tracker, frame_image):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true и воспользуйтесь нижним закомментированным кодом в этом файле для первого прогона, 
    на повторном прогоне можете читать сохраненные фреймы из папки
    и по координатам вырезать необходимые регионы.
    """
    detections = el['data']
    frame_id = el['frame_id']
    el['data'] = tracker.process_frame(detections, frame_id, frame_image)
    
    return el


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     print('Accepting client connection...')
#     await websocket.accept()
#     await websocket.send_text(str(country_balls))
#     tracker = Tracker()
#     data = []
#     for el in track_data:
#         print(el)
#         await asyncio.sleep(0.5)
#         # el = tracker_soft(el, tracker)
#         # data.append(el)
        
#         el = process_tracker_strong(el, tracker, frame_image)
#         data.append(el)

#         # el = tracker_strong(el)
#         # json_el = json.dumps(el)

#         await websocket.send_json(el)
#     metric = metric_fn(data)
#     print('Metric:', metric)
#     print('Bye..')



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    await websocket.receive_text()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте

    dir = "save_frames_dir"
    if not os.path.exists(dir):
        os.makedirs(dir)

    tracker = TrackerStrong()

    data = []
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        image_data = await websocket.receive_text()
        # print(image_data)
        try:
            image_data = re.sub('^data:image/.+;base64,', '', image_data)
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            image = image.resize((1000, 800), Image.Resampling.LANCZOS)
            frame_id = el['frame_id'] - 1
            image.save(f"{dir}/frame_{frame_id}.png")
            # print(image)
        except Exception as e:
            print(e)

        np_image = np.array(image)[...,:3]
        np_image = np.moveaxis(np_image, -1, 0)
        print(np_image.shape)
        el = process_tracker_strong(el, tracker, np_image)
        data.append(el)

        # el = tracker_strong(el)
        # json_el = json.dumps(el)

        await websocket.send_json(el)

        # отправка информации по фрейму
        await websocket.send_json(el)

    await websocket.send_json(el)
    await asyncio.sleep(0.5)
    image_data = await websocket.receive_text()
    try:
        image_data = re.sub('^data:image/.+;base64,', '', image_data)
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        image = image.resize((1000, 800), Image.Resampling.LANCZOS)
        image.save(f"{dir}/frame_{el['frame_id']}.png")
    except Exception as e:
        print(e)

    metric = metric_fn(data)
    print('Metric:', metric)
    print('Bye..')
