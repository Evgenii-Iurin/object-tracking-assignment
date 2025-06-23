## Tracker Soft

### Overview

To implement the **soft tracker**, we used:

* The Hungarian algorithm
* Kalman filter

**Input data:**

* The center of bounding boxes
* Bounding boxes from the detection algorithm used to calculate IoU

### Hungarian Algorithm

Initially, we implemented only the Hungarian algorithm, which solved the assignment problem.
We used the IoU between the bounding box from the previous frame and the current frame as the "cost" (cost matrix).

This approach has several drawbacks:

1. If the detector fails for several consecutive frames, the current bounding box no longer has a relevant "match" for comparison. We end up computing IoU between the current bounding box and the one from frame (t–N), which are unlikely to overlap. As a result, the entire track may be removed due to too many misses.
2. Track mixing. When objects intersect, the cost matrix might be constructed in a way that causes tracks to be incorrectly assigned to different objects. We visualized such a situation.

![tracks\_mixing](imgs/using_kalman_filter_in_tracking.gif)

> Note: This method is not a full-fledged tracker — it does not take into account other factors like velocity.

### Kalman Filter

To address the above issues, we used the Kalman filter.
A 2D motion transition matrix was used with a time step of Δt = 1.
Observations included only the coordinates of the bounding box center.
The filter's predictions were used to smooth and remove noisy detections that didn't match the expected trajectory.

In other words, the predicted position (center) of each active track is calculated first.
Then the distance between this position and each new detection center is computed.
If the distance is less than a given threshold (20 pixels), the IoU between the predicted and actual bounding boxes is calculated.
Based on the resulting cost matrix, the Hungarian algorithm is applied to find the best matches between tracks and new detections.

**Potential risks:**

* The object may move very erratically, rapidly changing speed.

### Metrics

For evaluation, we used **precision**:

* If the predicted track ID matches the ground truth ID, `correct += 1`
* Otherwise, `correct += 0`

`metric = correct / number of frames with detections (bounding box != [])`

### Tracker (Key Points)

* A new track is created if a detected object is not matched to any existing track.
* A track is removed if it hasn’t been updated for more than 10 consecutive frames (`missed > 10`)
* Euclidean distance and IoU are used to assess correspondence between prediction and detection
* Missed frame counter increases for unmatched tracks (`missed += 1`)
* Detections without a bounding box are filtered out and added back to the final output without tracking

## Results Table

| # | Filename                | Tracks Amount | Random Range | BB Skip Percent | Metric SOFT | Metric STRONG |
| - | ----------------------- | ------------- | ------------ | --------------- | ----------- | ------------- |
| 1 | tracks\_t5\_r0\_s0.py   | 5             | 0            | 0.0             | 0.9838      | 1.0000        |
| 2 | tracks\_t5\_r2\_s25.py  | 5             | 2            | 0.25            | 0.6752      | 0.2136        |
| 3 | tracks\_t5\_r5\_s50.py  | 5             | 5            | 0.5             | 0.4791      | 0.4166        |
| 4 | tracks\_t10\_r0\_s0.py  | 10            | 0            | 0.0             | 0.9114      | 0.5685        |
| 5 | tracks\_t10\_r2\_s25.py | 10            | 2            | 0.25            | 0.2346      | 0.3538        |
| 6 | tracks\_t10\_r5\_s50.py | 10            | 5            | 0.5             | 0.2095      | 0.0898        |
| 7 | tracks\_t20\_r0\_s0.py  | 20            | 0            | 0.0             | 0.9494      | 0.5898        |
| 8 | tracks\_t20\_r2\_s25.py | 20            | 2            | 0.25            | 0.1122      | 0.0489        |
| 9 | tracks\_t20\_r5\_s50.py | 20            | 5            | 0.5             | 0.0743      | 0.0743        |
