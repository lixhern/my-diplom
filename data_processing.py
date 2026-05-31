import os
import json
import shutil

root_images_folder = "rootData\Hemangiomas"
root_masks_folder = "project-11-at-2024-09-13-14-37-2642aa96"
annotations_json = "project-11-at-2024-09-13-14-37-2642aa96.json"
output_folder = "dataset"

os.makedirs(output_folder, exist_ok=True)

file_index = {}

for root, dirs, files in os.walk(root_images_folder):
    for file in files:
        if file.lower().endswith(".jpg"):
            file_index[file] = os.path.join(root, file)

print(f"[INFO] Images found: {len(file_index)}")

with open(annotations_json, "r", encoding="utf-8") as f:
    annotations_data = json.load(f)

print(f"[INFO] JSON items: {len(annotations_data)}")

success_count = 0
skipped_count = 0

for item in annotations_data:

    try:
        task_id = item["id"]

        raw_file = item.get("file_upload")
        original_filename = os.path.basename(raw_file).split("-", 1)[-1]

        if original_filename not in file_index:
            skipped_count += 1
            continue

        image_path = file_index[original_filename]

        annotations = item.get("annotations", [])
        if not annotations:
            skipped_count += 1
            continue

        annotation_id = annotations[0]["id"]

        mask_name_h = f"task-{task_id}-annotation-{annotation_id}-by-2-tag-Hemangioma-0.png"
        mask_name_l = f"task-{task_id}-annotation-{annotation_id}-by-2-tag-Liver-0.png"

        hemangioma_mask_path = os.path.join(root_masks_folder, mask_name_h)
        liver_mask_path = os.path.join(root_masks_folder, mask_name_l)

        if not os.path.exists(hemangioma_mask_path) or not os.path.exists(liver_mask_path):
            skipped_count += 1
            continue

        task_output_dir = os.path.join(output_folder, str(task_id))
        os.makedirs(task_output_dir, exist_ok=True)

        shutil.copy(image_path, os.path.join(task_output_dir, "original_image.jpg"))
        shutil.copy(hemangioma_mask_path, os.path.join(task_output_dir, "hemangioma_mask.png"))
        shutil.copy(liver_mask_path, os.path.join(task_output_dir, "liver_mask.png"))

        success_count += 1

    except Exception:
        skipped_count += 1

print(f"[RESULT] success: {success_count}")
print(f"[RESULT] skipped: {skipped_count}")