import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Define the base directory where the patient folders are located
base_dir = r"D:\Chicago_study\all_ARAT_videos"
# Assuming the CSV has columns 'Patient_ID' and 'Activity_ID'
csv_path = "D:/Tamim_deep_learning/ARAT_impairment/Segmentation/missing_filenames.csv"  # Replace with the path to your CSV file
# Define the output directory for saving CSV files
output_base_dir = r"D:\Tamim_deep_learning\ARAT_impairment\Segmentation\missing object files\data_res_trident\alternative\ipsilateral_0.85"

df = pd.read_csv(csv_path)

# Function to find the correct video path based on the criteria
def find_video_path(patient_id, activity_id):
    # Construct the patient folder path
    patient_folder = os.path.join(base_dir, patient_id)
    
    # Check if the patient folder exists
    if not os.path.exists(patient_folder):
        return None  # Patient folder not found
    
    # Iterate through all files in the patient folder
    for filename in os.listdir(patient_folder):
        # Check if the file is an MP4 video and matches the naming convention
        if filename.endswith(".mp4") and patient_id in filename and activity_id in filename:
            # Extract components from the filename
            parts = filename.split("_")
            arm_side = parts[2]  # 'left' or 'right'
            impairment_status = parts[3]  # 'Unimpaired' or 'Impaired'
            camera = parts[4]  # 'cam1' or 'cam4'
            
            # Check if the video is Impaired and matches the ipsi camera criteria
            if impairment_status == "Impaired":
                if (arm_side == "right" and camera == "cam1") or (arm_side == "left" and camera == "cam4"):
                    # Return the full path to the video
                    return os.path.join(patient_folder, filename)
    
    # If no matching video is found
    return None

# Function to process a video and save the tracking data
def process_video(video_path, output_csv_path):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error reading video: {video_path}")
        return
    
    # Let the user manually select an object in the first frame
    bbox = cv2.selectROI("Initial Bounding Box", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Initial Bounding Box")
    
    # Check if a valid bounding box was drawn
    if bbox == (0, 0, 0, 0):
        print("No bounding box selected. Saving NaN values.")
        tracker = None
    else:
        tracker = cv2.legacy.TrackerCSRT_create()  # Or cv2.TrackerCSRT_create() if using non-legacy
        tracker.init(frame, bbox)
    
    # Initialize lists to store object positions
    x_positions = []
    y_positions = []
    
    # Create CSV file
    with open(output_csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y"])  # Header row
        
        frame_count = 0  # Frame index
        
        # Start tracking loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if tracker:
                # Update tracker on every frame
                success, bbox = tracker.update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    center_x, center_y = x + w // 2, y + h // 2
                else:
                    center_x, center_y = np.nan, np.nan
            else:
                center_x, center_y = np.nan, np.nan
            
            # Store data
            x_positions.append(center_x)
            y_positions.append(center_y)
            
            # Write to CSV
            writer.writerow([center_x, center_y])
            
            # Only display (and pause) every 4th frame
            if frame_count % 4 == 0:
                # Draw bounding box if tracking is active and current coords are valid
                if not np.isnan(center_x):
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracked Object", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Object Tracking (showing every 4 frames)", frame)
                
                # Wait for user input
                key = cv2.waitKey(0)  # Pause until user presses a key
                if key == 27:  # ESC key to exit
                    break
                elif key == ord('r'):
                    # Reselect bounding box
                    bbox = cv2.selectROI("Correct Bounding Box", frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Correct Bounding Box")
                    if bbox != (0, 0, 0, 0):
                        tracker = cv2.legacy.TrackerCSRT_create()  # Or cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
    
    cap.release()
    cv2.destroyAllWindows()
    # Plot the y-coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(y_positions, label="Y Coordinate")
    plt.title(f"Y Coordinate Tracking for {os.path.basename(video_path)}")
    plt.xlabel("Frame Number")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid()
    plt.show()
# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    patient_id = row["Patient_ID"]
    activity_id = row["Activity_ID"]
    
    # Find the video path
    video_path = find_video_path(patient_id, activity_id)
    if not video_path:
        print(f"No matching video found for Patient ID: {patient_id}, Activity ID: {activity_id}")
        continue
    
    # Create the output directory for the patient
    patient_output_dir = os.path.join(output_base_dir, patient_id)
    os.makedirs(patient_output_dir, exist_ok=True)
    
    # Generate the output CSV filename
    video_filename = os.path.basename(video_path)
    parts = video_filename.split("_")
    output_csv_filename = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_ipsi_{activity_id}.csv"
    output_csv_path = os.path.join(patient_output_dir, output_csv_filename)
    
    # Process the video and save the tracking data
    print(f"Processing video: {video_path}")
    process_video(video_path, output_csv_path)
    print(f"Saved tracking data to: {output_csv_path}")

