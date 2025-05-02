import logging 
import numpy as np
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO)

class Track():
    def __init__(self, track_id, bounding_box, frame_id):
        self.track_id = track_id
        self.bounding_box = bounding_box
        self.frame_id = frame_id

class Tracker():
    def __init__(self):
        self.track_id = 0
        self.tracks = []
    
    def create_track(self, bounding_box, frame_id):
        """
        Create a new track with the given bounding box and frame ID
        """
        track = Track(self.track_id, bounding_box, frame_id)
        self.tracks.append(track)
        self.track_id += 1
        return track
    
    def process_frame(self, detections, frame_id):
        """
        Process current frame and update tracks
        
        :param detections:
            List of dictionaries with the following keys
                - cb_id: id of the country ball (grand truth - is not used)
                - bounding_box: [x1, y1, x2, y2]
                - x: x coordinate of the center of the bounding box
                - y: y coordinate of the center of the bounding box
                - track_id: id of the track (None if not assigned)
                
                e.g. {'cb_id': 0, 'bounding_box': [], 'x': 1000, 'y': 519, 'track_id': None}
        :param frame_id: number of the current frame
        """
        if len(self.tracks) == 0:
            # If there are no tracks, create a new track for each detection
            for detection in detections:
                track = self.create_track(detection['bounding_box'], frame_id)
                detection['track_id'] = track.track_id
                
            return detections
        
        # Build the IoU matrix between tracks and detections
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        # Calculate IoU for each track and detection
        for i, detection in enumerate(detections):
            for j, track in enumerate(self.tracks):
                iou = self.calculate_iou(track.bounding_box, detection['bounding_box'])
                logging.info("IOU for %d, %d: %f", i, j, iou)
                iou_matrix[i,j] = 1 - iou

        detection_indxs, tracks_indxs = linear_sum_assignment(iou_matrix)

        for detection_ids, tracks_ids in zip(detection_indxs, tracks_indxs):
            detection = detections[detection_ids]
            track = self.tracks[tracks_ids]

            # Update the track with the new bounding box
            track.bounding_box = detection['bounding_box']
            track.frame_id = frame_id
            
            detection['track_id'] = track.track_id
            
        return detections



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
