import numpy as np
import random
from scipy.optimize import curve_fit
import os

# Параметры для генерации
experiments_params = [
    {'tracks_amount': 5, 'random_range': 0, 'bb_skip_percent': 0.0}
]

width = 1000
height = 800
cb_width = 120
cb_height = 100

os.makedirs("experiments", exist_ok=True)

def get_point_on_random_side(width, height):
    side = random.randint(0, 4)
    if side == 0:
        x = random.randint(0, width)
        y = 0
    elif side == 1:
        x = random.randint(0, width)
        y = height
    elif side == 2:
        x = 0
        y = random.randint(0, height)
    else:
        x = width
        y = random.randint(0, height)
    return x, y

def fun(x, a, b, c, d):
    return a * x + b * x ** 2 + c * x ** 3 + d

def check_track(track):
    if all(el['x'] == track[0]['x'] for el in track):
        return False
    if all(el['y'] == track[0]['y'] for el in track):
        return False
    if not all(0 <= el['x'] <= width for el in track):
        return False
    if not all(0 <= el['y'] <= height for el in track):
        return False
    return True

def add_track_to_tracks(track, tracks, bb_skip_percent, random_range, id):
    for i, p in enumerate(track):
        if random.random() < bb_skip_percent:
            bounding_box = []
        else:
            bounding_box = [
                p['x'] - int(cb_width / 2) + random.randint(-random_range, random_range),
                p['y'] - cb_height + random.randint(-random_range, random_range),
                p['x'] + int(cb_width / 2) + random.randint(-random_range, random_range),
                p['y'] + random.randint(-random_range, random_range)
            ]
        if i < len(tracks):
            tracks[i]['data'].append({'cb_id': id, 'bounding_box': bounding_box,
                                      'x': p['x'], 'y': p['y'], 'track_id': None})
        else:
            tracks.append({
                'frame_id': len(tracks) + 1,
                'data': [{'cb_id': id, 'bounding_box': bounding_box,
                          'x': p['x'], 'y': p['y'], 'track_id': None}]
            })
    return tracks

# Генерация .py файлов
for params in experiments_params:
    tracks_amount = params['tracks_amount']
    random_range = params['random_range']
    bb_skip_percent = params['bb_skip_percent']

    tracks = []
    i = 0

    while i < tracks_amount:
        x, y = np.array([]), np.array([])
        p = get_point_on_random_side(width, height)
        x = np.append(x, p[0])
        y = np.append(y, p[1])
        x = np.append(x, random.randint(200, width - 200))
        y = np.append(y, random.randint(200, height - 200))
        x = np.append(x, random.randint(200, width - 200))
        y = np.append(y, random.randint(200, height - 200))
        p = get_point_on_random_side(width, height)
        x = np.append(x, p[0])
        y = np.append(y, p[1])
        num = random.randint(20, 50)

        try:
            coef, _ = curve_fit(fun, x, y)
            track = [{'x': int(x), 'y': int(y)} for x, y in
                     zip(np.linspace(x[0], x[-1], num=num), fun(np.linspace(x[0], x[-1], num=num), *coef))]
            if check_track(track):
                tracks = add_track_to_tracks(track, tracks, bb_skip_percent, random_range, i)
                i += 1
        except:
            continue

    # Формируем имя и сохраняем .py файл
    fname = f"tracks_t{tracks_amount}_r{random_range}_s{int(bb_skip_percent*100)}.py"
    fpath = os.path.join("experiments", fname)

    with open(fpath, 'w') as f:
        f.write(f"# Автоматически сгенерированный файл\n")
        f.write(f"country_balls_amount = {tracks_amount}\n")
        f.write(f"track_data = {repr(tracks)}\n")

    print(f"[✓] Saved: {fpath}")
