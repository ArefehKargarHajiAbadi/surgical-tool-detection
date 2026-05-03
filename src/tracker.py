from boxmot import BoTSORT
import numpy as np

class ObjectTracker:
    def __init__(self, track_high_conf=True):
        self.tracker = BoTSORT(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=0.8
        )
        self.track_id = 0

    def update(self, detections: list, frame: np.ndarray):
        """Update tracker and return tracks with IDs."""
        if not detections:
            return []
        # Convert to format expected by boxmot: [x1,y1,x2,y2,conf,cls]
        dets = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
                          d["confidence"], d["class"]] for d in detections])
        tracks = self.tracker.update(dets, frame)
        # tracks: [x1,y1,x2,y2,track_id,conf,cls]
        results = []
        for t in tracks:
            results.append({
                "bbox": t[:4],
                "track_id": int(t[4]),
                "confidence": t[5],
                "class": int(t[6])
            })
        return results