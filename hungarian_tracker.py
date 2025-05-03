import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from hungarian_algorithm import solve_hungarian_algorithm

logging.basicConfig(level=logging.INFO)

class Track():
    def __init__(self, track_id, bounding_box, frame_id):
        self.track_id = track_id
        self.bounding_box = bounding_box
        self.frame_id = frame_id
        self.missed = 0

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
        logging.info("Created new track with ID: %d", track.track_id)
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
                if len(detection['bounding_box']) == 0:
                    continue
                track = self.create_track(detection['bounding_box'], frame_id)
                detection['track_id'] = track.track_id
                
            return detections
        
        # Filter out empty detections
        detections_without_bboxes = [detection for detection in detections if len(detection['bounding_box']) == 0]
        detections = [detection for detection in detections if len(detection['bounding_box']) > 0]

        # Build the IoU matrix between tracks and detections
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        # ========== Calculate IoU for each track and detection pair ==========
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):

                if len(detection['bounding_box']) == 0:
                    iou_matrix[i,j] = 1e6
                else:
                    iou = self.calculate_iou(track.bounding_box, detection['bounding_box'])
                    iou_matrix[i, j] = (1 - iou) ** 2
                    logging.info("Frame: %d : IOU for %d, %d: %f", frame_id, i, j, iou)


        # ========== Solve the assignment problem using Hungarian algorithm ==========

        matched_tracks_idxs, matched_detections_idxs  = solve_hungarian_algorithm(iou_matrix)
        for track_idx, detection_idx in zip(matched_tracks_idxs, matched_detections_idxs):
            detection = detections[detection_idx]
            track = self.tracks[track_idx]

            # Update the track with the new bounding box
            track.bounding_box = detection['bounding_box']
            track.frame_id = frame_id
            
            detection['track_id'] = track.track_id

        # ========== Filter out old tracks ==================

        matched_track_indices_set = set(matched_tracks_idxs)
        for i, track in enumerate(self.tracks):
            if i in matched_track_indices_set:
                track.missed = 0
            else:
                track.missed += 1
                
        # Filter all tracks that have been missed more than a certain threshold
        num_missed = 10
        self.tracks = [track for track in self.tracks if track.missed < num_missed]
        
        logging.info("Frame %d : Tracks after filtering: %d", frame_id, len(self.tracks))            
    
        # ====== Create new tracks for unmatched detections ======
        unmatched_detections = [detection for i, detection in enumerate(detections) if i not in matched_detections_idxs]
        for detection in unmatched_detections:
            if len(detection['bounding_box']) == 0:
                continue
            # Create a new track for the unmatched detection
            track = self.create_track(detection['bounding_box'], frame_id)
            detection['track_id'] = track.track_id

        logging.info("Frame %d : Tracks after creating new ones: %d", frame_id, len(self.tracks))

        detections = detections + detections_without_bboxes
            
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
