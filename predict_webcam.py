import cv2
from ultralytics import YOLO

# 1. Load your local model weights
model_path = "best_fruits_veggies_model.pt"
model = YOLO(model_path)

# 2. Connect directly to your local webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access your webcam device.")
    exit()

# Set camera capture properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📸 Webcam running at full resolution!")
print("👉 Press 'q' on the popup window to stop the stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Warning: Failed to grab webcam frame.")
        break

    # CHANGES FOR MAXIMUM ACCURACY:
    # - Removed 'imgsz=480': Let it default to full 640 resolution
    # - Lowered 'conf' to 0.25: Forces the model to show boxes even if it is only 25% sure
    # - Removed 'half=True': Run at full precision (FP32) for better feature extraction
    results = model.predict(
        frame, 
        conf=0.25, 
        verbose=False
    )
    
    # Plot the box overlays
    annotated_frame = results[0].plot()

    # Display the native window output
    cv2.imshow("YOLO26 High-Accuracy Webcam Detections", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("👋 Webcam stream stopped cleanly.")