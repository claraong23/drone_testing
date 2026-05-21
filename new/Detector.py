import os
import cv2
import time
import queue
import threading
import numpy as np
from typing import Callable, Optional, Any, List, Dict
from ultralytics import YOLO

class Detector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        callback: Optional[Callable[[List[Dict[str, Any]], np.ndarray, Optional[Any]], None]] = None,
        num_workers: int = 1,
        device: str = "cpu",
        save_dir: str = "./detected_images",
        enable_display: bool = True,
        display_window_name: str = "YOLO Detections"
    ):
        self.model = YOLO(model_path).to(device)
        self.conf_threshold = confidence_threshold
        self.callback = callback
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.workers = []
        
        # Setup save directory
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self._file_counter = 0
        self._counter_lock = threading.Lock()
        
        # Display configuration
        self.enable_display = enable_display
        self.display_window_name = display_window_name
        # Maxsize=1 ensures we always show the LATEST frame, dropping old ones to prevent UI lag
        self.display_queue = queue.Queue(maxsize=1)
        self.display_thread = None
        
        self._start_workers(num_workers)
        
        if self.enable_display:
            self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
            self.display_thread.start()

    def _start_workers(self, num_workers: int) -> None:
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker, daemon=False)
            t.start()
            self.workers.append(t)

    def submit_image(self, image: np.ndarray, context: Optional[Dict[str, Any]] = None) -> None:
        if context is None:
            context = {}
        self.queue.put((image, context))

    def _get_next_filename(self) -> str:
        with self._counter_lock:
            self._file_counter += 1
            ts = int(time.time() * 1000)
            return f"det_{self._file_counter}_{ts}.jpg"

    def _display_worker(self) -> None:
        """Dedicated thread for rendering detected images."""
        cv2.namedWindow(self.display_window_name, cv2.WINDOW_AUTOSIZE)
        while not self.stop_event.is_set():
            try:
                img = self.display_queue.get(timeout=0.05)
                if img is not None:
                    cv2.imshow(self.display_window_name, img)
                    cv2.waitKey(1)  # Mandatory for OpenCV GUI event processing
            except queue.Empty:
                continue
            except cv2.error as e:
                print(f"[Display] CV2 error: {e}")
                break
        cv2.destroyWindow(self.display_window_name)

    def _worker(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                image, context = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                results = self.model(image, verbose=False, conf=self.conf_threshold)
                
                detections = []
                annotated_image = None
                has_detections = False

                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        has_detections = True
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                            conf = float(box.conf[0].cpu().item())
                            cls_id = int(box.cls[0].cpu().item())
                            
                            detections.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": conf,
                                "class_id": cls_id,
                                "class_name": self.model.names[cls_id]
                            })
                        annotated_image = result.plot()

                if has_detections and annotated_image is not None:
                    # 1. Save to disk
                    filename = self._get_next_filename()
                    filepath = os.path.join(self.save_dir, filename)
                    cv2.imwrite(filepath, annotated_image)
                    context["saved_path"] = filepath

                    # 2. Push to display thread (non-blocking: drops old frames if UI is slow)
                    if self.enable_display:
                        try:
                            self.display_queue.put_nowait(annotated_image)
                        except queue.Full:
                            pass  # Skip frame to keep display real-time

                    # 3. Trigger callback
                    if self.callback:
                        try:
                            self.callback(detections, annotated_image, context)
                        except Exception as e:
                            print(f"[Detector] Callback error: {e}")

            except Exception as e:
                print(f"[Detector] Inference error: {e}")
            finally:
                self.queue.task_done()
                # Explicit memory cleanup as requested
                del image
                del context
                del results
                del detections
                del annotated_image

    def stop(self) -> None:
        """Gracefully shutdown all threads."""
        self.stop_event.set()
        for t in self.workers:
            t.join()
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join()
        print(f"[Detector] All workers and display thread stopped.")

    def set_display(self, enabled: bool) -> None:
        """Toggle display at runtime."""
        self.enable_display = enabled