from djitellopy import Tello
import os
import cv2
import time
import sys
from pathlib import Path
import csv
from datetime import datetime

# Add yolov7 to Python path
yolov7_path = Path("yolov7")
sys.path.append(str(yolov7_path))
from yolov7.detector import YoloDetector

# Configure FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

def connect_tello(retries=3, delay=2):
    """Robust Tello connection with retries"""
    tello = Tello()
    for attempt in range(retries):
        try:
            print(f"[STATUS] Connection attempt {attempt+1}/{retries}")
            tello.connect()
            print(f"[SUCCESS] Connected! Battery: {tello.get_battery()}%")
            return tello
        except Exception as e:
            print(f"[WARNING] Attempt {attempt+1} failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    raise ConnectionError("Failed to connect to Tello after multiple attempts")

def main():
    # Initialize metrics
    metrics = {
        'frames_processed': 0,
        'total_inference': 0,
        'total_processing': 0,
        'start_time': time.time()
    }
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"tello_perf_{timestamp}.csv"
    
    try:
        # Connect to Tello with error handling
        tello = connect_tello()
        
        # Start video stream
        tello.streamon()
        frame_reader = tello.get_frame_read()
        time.sleep(2)  # Stream stabilization

        # Initialize YOLO
        print("[STATUS] Loading YOLOv7...")
        detector = YoloDetector("yolov7/yolov7-tiny.pt")
        
        # Video writer
        video_out = cv2.VideoWriter(
            f"output_{timestamp}.avi", 
            cv2.VideoWriter_fourcc(*'MJPG'), 
            20, (960, 720)
        )

        # CSV writer
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'frame', 'inference_ms', 'processing_ms', 
                'fps', 'persons', 'battery'
            ])

            print("[STATUS] Starting detection loop (Press Q to quit)")
            while True:
                frame_start = time.time()
                
                # Capture frame
                frame = frame_reader.frame
                if frame is None:
                    print("[WARNING] Empty frame")
                    continue
                    
                frame = cv2.resize(frame, (960, 720))
                
                # Detection
                inf_start = time.time()
                results = detector.detect(frame)
                inf_time = time.time() - inf_start
                
                # Process results
                persons = 0
                for res in results:
                    if res["label"] == "person":
                        persons += 1
                        x1, y1, x2, y2 = res["box"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                
                # Calculate metrics
                proc_time = time.time() - frame_start
                fps = 1.0 / proc_time
                
                # Update totals
                metrics['frames_processed'] += 1
                metrics['total_inference'] += inf_time
                metrics['total_processing'] += proc_time
                
                # Write to CSV
                writer.writerow([
                    metrics['frames_processed'],
                    round(inf_time*1000, 2),
                    round(proc_time*1000, 2),
                    round(fps, 1),
                    persons,
                    tello.get_battery()
                ])
                
                # Display info
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                cv2.putText(frame, f"Persons: {persons}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                
                video_out.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"[ERROR] {str(e)}")
    finally:
        # Calculate summary only if frames processed
        if metrics['frames_processed'] > 0:
            total_time = time.time() - metrics['start_time']
            print("\n=== PERFORMANCE SUMMARY ===")
            print(f"Frames processed: {metrics['frames_processed']}")
            print(f"Average FPS: {metrics['frames_processed']/total_time:.1f}")
            print(f"Avg inference: {metrics['total_inference']/metrics['frames_processed']*1000:.1f}ms")
            print(f"CSV log saved to: {csv_file}")
        
        # Cleanup
        if 'tello' in locals():
            tello.streamoff()
        if 'video_out' in locals():
            video_out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
