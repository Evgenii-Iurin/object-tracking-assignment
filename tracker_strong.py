import logging
import numpy as np
from scipy.spatial.distance import cdist
from hungarian_algorithm import solve_hungarian_algorithm
from kalman_filter import KalmanFilter2D
import torch
import torchvision.models as models
import torchvision.transforms as transforms

logging.basicConfig(level=logging.INFO)

class Track():
    def __init__(self, track_id, bounding_box, cx, cy, frame_id, appearance_feature):
        self.track_id = track_id
        self.bounding_box = bounding_box
        self.frame_id = frame_id
        self.missed = 0
        self.kalman = KalmanFilter2D(cx, cy)
        self.features = [appearance_feature]  # Храним историю фич
        self.max_features = 100  # Максимальное количество сохраняемых фич

    def predict(self):
        predicted_center = self.kalman.predict()
        w = self.bounding_box[2] - self.bounding_box[0]
        h = self.bounding_box[3] - self.bounding_box[1]
        
        predicted_box = [
            predicted_center[0] - w/2,
            predicted_center[1] - h/2,
            predicted_center[0] + w/2,
            predicted_center[1] + h/2
        ]
        
        return predicted_center, predicted_box

    def update(self, bounding_box, cx, cy, appearance_feature):
        self.bounding_box = bounding_box
        self.kalman.update([cx, cy])
        self.features.append(appearance_feature)
        if len(self.features) > self.max_features:
            self.features.pop(0)

class FeatureExtractor:
    def __init__(self):
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image_patch):
        if image_patch.size == 0 or image_patch.shape[0] < 5 or image_patch.shape[1] < 5:
            return np.zeros(1280)  # Возвращаем нулевой вектор при ошибке
        
        try:
            # Конвертируем BGR в RGB если используем OpenCV
            if image_patch.shape[2] == 3:
                image_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
            
            # Предобработка и нормализация
            tensor = self.preprocess(image_patch).unsqueeze(0)
            
            with torch.no_grad():
                features = self.model.features(tensor)
                return torch.nn.functional.adaptive_avg_pool2d(features, 1).squeeze().numpy()
        
        except Exception as e:
            logging.error(f"Feature extraction error: {str(e)}")
            return np.zeros(1280)


class TrackerStrong():
    def __init__(self):
        self.track_id = 0
        self.tracks = []
        self.feature_extractor = FeatureExtractor()
        self.lambda_ = 0.5  # Вес для фич-эмбеддингов
        self.max_misses = 10
    
    def create_track(self, bounding_box, cx, cy, frame_id, appearance_feature):
        track = Track(self.track_id, bounding_box, cx, cy, frame_id, appearance_feature)
        self.tracks.append(track)
        self.track_id += 1
        logging.info("Created new track with ID: %d", track.track_id)
        return track
    
    def process_frame(self, detections, frame_id, frame_image):
        # Фильтрация пустых детекций
        detections_without_bboxes = [d for d in detections if len(d['bounding_box']) == 0]
        valid_detections = [d for d in detections if len(d['bounding_box']) > 0]

        # Инициализация треков при первом кадре
        if not self.tracks:
            for det in valid_detections:
                patch = self._get_patch(frame_image, det['bounding_box'])
                feature = self.feature_extractor(patch)
                self.create_track(
                    det['bounding_box'],
                    det['x'],
                    det['y'],
                    frame_id,
                    feature
                )
                det['track_id'] = self.track_id - 1
            return detections

        # 1. Предсказание позиций треков
        predicted_boxes = []
        predicted_centers = []
        for track in self.tracks:
            center, box = track.predict()
            predicted_centers.append(center)
            predicted_boxes.append(box)

        # 2. Расчет матриц стоимости
        num_tracks = len(self.tracks)
        num_detections = len(valid_detections)
        
        # Инициализация матриц
        motion_cost = np.full((num_tracks, num_detections), 1e6)
        appearance_cost = np.full((num_tracks, num_detections), 1.0)
        
        # 2.1 Расчет стоимости движения
        for i, track in enumerate(self.tracks):
            track_center = predicted_centers[i]
            cov = track.kalman.P[:2, :2]
            
            for j, det in enumerate(valid_detections):
                det_center = np.array([det['x'], det['y']])
                
                # Расстояние Махаланобиса
                try:
                    inv_cov = np.linalg.pinv(cov)
                    motion_dist = np.sqrt((det_center - track_center).T @ inv_cov @ (det_center - track_center))
                except:
                    motion_dist = 1e6
                    
                # Евклидово расстояние как fallback
                euclidean_dist = np.linalg.norm(det_center - track_center)
                motion_cost[i,j] = min(motion_dist, euclidean_dist)

        # 2.2 Расчет стоимости внешнего вида
        det_features = []
        for j, det in enumerate(valid_detections):
            patch = self._get_patch(frame_image, det['bounding_box'])
            feature = self.feature_extractor(patch)
            det_features.append(feature / (np.linalg.norm(feature) + 1e-6))

        for i, track in enumerate(self.tracks):
            track_feats = np.array([f / (np.linalg.norm(f) + 1e-6) for f in track.features])
            mean_track_feat = track_feats.mean(axis=0)
            
            for j, det_feat in enumerate(det_features):
                appearance_cost[i,j] = 1.0 - np.dot(mean_track_feat, det_feat)

        # 3. Комбинированная матрица стоимости
        cost_matrix = self.lambda_ * motion_cost + (1 - self.lambda_) * appearance_cost
        
        # 4. Обработка особых случаев
        cost_matrix = np.nan_to_num(cost_matrix, nan=1e6)
        cost_matrix = np.clip(cost_matrix, 0, 1e6)

        # 5. Решение задачи назначения
        track_indices, det_indices = solve_hungarian_algorithm(cost_matrix)

        # 6. Обновление совпавших треков
        updated_tracks = set()
        for t_idx, d_idx in zip(track_indices, det_indices):
            track = self.tracks[t_idx]
            det = valid_detections[d_idx]
            
            patch = self._get_patch(frame_image, det['bounding_box'])
            feature = self.feature_extractor(patch)
            
            track.update(
                det['bounding_box'],
                det['x'],
                det['y'],
                feature
            )
            det['track_id'] = track.track_id
            updated_tracks.add(t_idx)

        # 7. Обработка неподходящих треков
        for i, track in enumerate(self.tracks):
            if i not in updated_tracks:
                track.missed += 1

        # 8. Удаление потерянных треков
        self.tracks = [t for t in self.tracks if t.missed < self.max_misses]

        # 9. Создание новых треков
        unmatched_detections = [j for j in range(num_detections) if j not in det_indices]
        for j in unmatched_detections:
            det = valid_detections[j]
            patch = self._get_patch(frame_image, det['bounding_box'])
            feature = self.feature_extractor(patch)
            
            new_track = self.create_track(
                det['bounding_box'],
                det['x'],
                det['y'],
                frame_id,
                feature
            )
            det['track_id'] = new_track.track_id

        # 10. Сборка финального результата
        final_detections = valid_detections + detections_without_bboxes
        final_detections = [x for x in final_detections 
                            if 'track_id' in x and x['track_id'] is not None]
        print(final_detections)
        return sorted(final_detections, key=lambda x: x.get('track_id', -1))

    def _get_patch(self, image, bbox):
        # Извлечение патча объекта из изображения
        x1, y1, x2, y2 = map(int, bbox)
        return image[y1:y2, x1:x2]

    
    def calculate_iou(self, box1, box2):
        """
        Вычисляет IoU (Intersection over Union) между двумя bounding box'ами.
        box1, box2: списки вида [x_min, y_min, x_max, y_max]
        """

        # Find coordinates of the intersection rectangle
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        # If there is no intersection, return 0
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - intersection_area

        iou = intersection_area / union_area

        return iou