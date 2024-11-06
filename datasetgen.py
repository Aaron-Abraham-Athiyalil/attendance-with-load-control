import os
import shutil

# Parameters
source_folder = f"C:/Users/Admin/Documents/GitHub/attendance-with-load-control/source_images"  # Folder where all images are currently stored
destination_folder = "C:/Users/Admin/Documents/GitHub/attendance-with-load-control/dataset"   # Root folder for organized dataset
num_people = 5                   # Number of people
images_per_person = 5            # Images to collect per person

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Organize images for each person
for person_id in range(1, num_people + 1):
    # Create a folder for each person
    person_folder = os.path.join(destination_folder, f"person{person_id}")
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    
    # Collect images for the person
    count = 0
    for filename in os.listdir(source_folder):
        if filename.startswith(f"person{person_id}_") and filename.endswith(".jpg"):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(person_folder, f"{count + 1}.jpg")
            
            # Move or copy the file to the destination folder
            shutil.copy(source_path, destination_path)  # Use shutil.move() to move instead of copy
            
            count += 1
            print(f"Copied {filename} to {destination_path}")
            
            # Stop after collecting the required number of images
            if count >= images_per_person:
                break

    print(f"Completed organizing images for person{person_id}")

print("Dataset organization complete.")
