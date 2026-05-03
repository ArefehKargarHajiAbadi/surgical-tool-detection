import cv2
import time
import logging
from .utils import draw_boxes, calculate_fps

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, detector, conf_thresh, iou_thresh, video_source):
        self.detector = detector
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.video_source = video_source
        self.cap = None
        self.writer = None

    def run(self):
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source {self.video_source}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = os.getenv("OUTPUT_VIDEO", None)
        if out_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            detections = self.detector.detect(frame, self.conf_thresh, self.iou_thresh)
            annotated = draw_boxes(frame, detections)
            if os.getenv("SHOW_FPS", "True").lower() == "true":
                elapsed = time.time() - start_time
                current_fps = calculate_fps(frame_count, elapsed)
                cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Surgical Tool Detection", annotated)
            if self.writer:
                self.writer.write(annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count += 1

        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()