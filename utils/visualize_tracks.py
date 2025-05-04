import json
import cv2
import numpy as np
import argparse
import os

def get_cb_id_color(cb_id):
    colors = {
        0: (0, 0, 255),    # Red
        1: (255, 0, 0),    # Blue
        2: (0, 255, 0),    # Green
        3: (0, 165, 255),  # Orange
    }
    return colors.get(cb_id, (255, 255, 255))  # White default

def get_track_id_color(track_id):
    colors = {
        0: (0, 0, 255),    # Red
        1: (255, 0, 0),    # Blue
        2: (0, 255, 0),    # Green
        3: (0, 165, 255),  # Orange
    }
    return colors.get(track_id, (100, 100, 100))  # Gray for unknown track_id

def draw_box_and_label(frame, bbox, cb_id, track_id):
    # Цвет bbox по cb_id (ground truth)
    box_color = get_cb_id_color(cb_id)
    # Цвет текста по track_id (prediction)
    label_color = get_track_id_color(track_id)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label_text = f"Track {track_id}" if track_id is not None else "Track None"
    cv2.putText(frame, label_text, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)

def make_tracking_video(json_path, output_path, frame_size=(1280, 720), fps=10):
    with open(json_path, 'r') as f:
        lines = f.readlines()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Траектории по cb_id
    trajectory_history = {}

    for line in lines:
        frame_data = json.loads(line)
        frame = np.ones((frame_size[1], frame_size[0], 3), dtype=np.uint8) * 255  # белый фон

        for obj in frame_data.get("data", []):
            bbox = obj.get("bounding_box", [])
            cb_id = obj.get("cb_id", None)
            track_id = obj.get("track_id", None)

            if bbox and cb_id is not None:
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(frame_size[0] - 1, int(bbox[2]))
                y2 = min(frame_size[1] - 1, int(bbox[3]))

                # Центр bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # История по cb_id
                if cb_id not in trajectory_history:
                    trajectory_history[cb_id] = []
                trajectory_history[cb_id].append((cx, cy))

                draw_box_and_label(frame, (x1, y1, x2, y2), cb_id, track_id)

        # Рисуем траектории по cb_id (ground truth)
        for history in trajectory_history.values():
            if len(history) > 1:
                for i in range(1, len(history)):
                    pt1 = history[i - 1]
                    pt2 = history[i]
                    cv2.line(frame, pt1, pt2, (0, 0, 0), 2)  # черная линия

        out.write(frame)

    out.release()
    print(f"✅ Видео успешно сохранено: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Визуализация треков: цвет bbox по cb_id, label по track_id")
    parser.add_argument("input", type=str, help="Путь к JSON-файлу с данными")
    parser.add_argument("output", type=str, help="Путь для сохранения выходного видео (mp4)")
    parser.add_argument("--width", type=int, default=1000, help="Ширина видеофрейма")
    parser.add_argument("--height", type=int, default=800, help="Высота видеофрейма")
    parser.add_argument("--fps", type=int, default=3, help="Кадров в секунду")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Файл не найден: {args.input}")
        return

    make_tracking_video(args.input, args.output, frame_size=(args.width, args.height), fps=args.fps)

if __name__ == "__main__":
    main()
