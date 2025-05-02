import logging

logging.basicConfig(level=logging.INFO)


def metric_fn(data) -> float:
    """
    :param data:
        List of dictionaries with the following keys
            - cb_id: id of the country ball (grand truth - is not used)
            - bounding_box: [x1, y1, x2, y2]
            - x: x coordinate of the center of the bounding box
            - y: y coordinate of the center of the bounding box
            - track_id: id of the track (None if not assigned)
            
            e.g. 
                {'cb_id': 0, 'bounding_box': [], 'x': 1000, 'y': 519, 'track_id': None}
                {'cb_id': 0, 'bounding_box': [13, 45, 190, -102], 'x': 50, 'y': 1800, 'track_id': None}
    :return: precision metric
    """
    
    correct = 0
    num_detections = 0
    
    for detections in data:
        for detection in detections['data']:
            if len(detection['bounding_box']) == 0:
                continue
            grand_truth = detection['cb_id']
            track_id = detection['track_id']
            correct += 1 if grand_truth == track_id else 0
            num_detections += 1
            
            logging.info("Detection: Grand Truth: %s, Track ID: %s", grand_truth, track_id)
            
            
    precision = correct / num_detections if num_detections > 0 else 0
    return precision
