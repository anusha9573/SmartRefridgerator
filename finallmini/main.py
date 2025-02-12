# import cv2
# import pymongo
# import supervision as sv
# from ultralytics import YOLO

# # MongoDB setup
# mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
# db = mongo_client["fridge"]
# collection = db["items"]

# # Check connection
# try:
#     mongo_client.admin.command("ping")
#     print("Successfully connected to MongoDB")
# except ConnectionError as e:
#     print("Failed to connect to MongoDB:", e)

# # Initialize YOLO model
# model = YOLO("best.pt")
# box_annotator = sv.BoxAnnotator()
# label_annotator = sv.LabelAnnotator()

# # Capture video from webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Unable to read camera feed")
#     exit()

# object_counts = {}
# previous_counts = {}

# # Assuming YOLO model provides class names through `model.names`
# class_names = model.names  # List of class names

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to capture frame from camera. Exiting...")
#         break

#     # Perform object detection
#     results = model(frame)
#     if not results:
#         print("No results returned from model")
#         continue

#     results = results[0]
#     detections = sv.Detections.from_ultralytics(results)

#     # Clear previous object counts
#     object_counts.clear()

#     # Count detected objects
#     for i in range(len(detections.class_id)):
#         class_id = detections.class_id[i]
#         object_label = class_names[class_id]  # Get class name from ID

#         if object_label in object_counts:
#             object_counts[object_label] += 1
#         else:
#             object_counts[object_label] = 1

#     # Check for changes in counts
#     count_changed = False
#     for object_label in object_counts:
#         if object_label in previous_counts:
#             if object_counts[object_label] != previous_counts[object_label]:
#                 count_changed = True
#                 break
#         else:
#             count_changed = True
#             break

#     # Initialize data variable
#     data = None

#     # Update MongoDB if there's a change in counts
#     if count_changed:
#         for object_label in object_counts:
#             # Fetch the current document from MongoDB
#             existing_item = collection.find_one({"name": object_label})

#             if existing_item:
#                 # Update existing document
#                 new_quantity = existing_item["quantity"] + object_counts[object_label]
#                 collection.update_one(
#                     {"name": object_label},
#                     {"$set": {"quantity": new_quantity}},
#                 )
#                 data = {
#                     "name": object_label,
#                     "quantity": new_quantity,
#                     "unit": existing_item.get("unit", None),
#                 }
#             else:
#                 # Insert new document if not found
#                 data = {
#                     "name": object_label,
#                     "quantity": object_counts[object_label],
#                     "unit": None,
#                 }
#                 collection.insert_one(data)

#             # Debugging: Print data being inserted/updated
#             print(f"Updating MongoDB with: {data}")

#         previous_counts = object_counts.copy()

#     # Annotate and display the frame with counts
#     annotated_image = frame.copy()
#     for i in range(len(detections.class_id)):
#         class_id = detections.class_id[i]
#         object_label = class_names[class_id]
#         count = object_counts.get(object_label, 0)

#         # Draw bounding box
#         box = detections.xyxy[i]
#         cv2.rectangle(
#             annotated_image,
#             (int(box[0]), int(box[1])),
#             (int(box[2]), int(box[3])),
#             (0, 255, 0),
#             2,
#         )

#         # Add label with count
#         label = f"{object_label}: {count}"
#         cv2.putText(
#             annotated_image,
#             label,
#             (int(box[0]), int(box[1]) - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 255, 0),
#             2,
#         )

#     cv2.imshow("Webcam", annotated_image)

#     if cv2.waitKey(1) % 256 == 27:
#         print("Escape hit, closing...")
#         break

# cap.release()
# cv2.destroyAllWindows()
# mongo_client.close()

import cv2
import pymongo
import supervision as sv
from ultralytics import YOLO

# MongoDB setup
mongo_client = pymongo.MongoClient("mongodb://localhost:27017")
db = mongo_client["fridge"]
collection = db["items"]

# Check connection
try:
    mongo_client.admin.command("ping")
    print("Successfully connected to MongoDB")
except ConnectionError as e:
    print("Failed to connect to MongoDB:", e)

# Initialize YOLO model
model = YOLO("best.pt")
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Capture video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to read camera feed")
    exit()

object_counts = {}
previous_counts = {}

# Initialize a dictionary to store the previous positions of detected objects
object_positions = {}

# Assuming YOLO model provides class names through `model.names`
class_names = model.names  # List of class names


def update_mongo_db(object_label, movement_direction, count):
    """Update MongoDB based on whether the object is added or removed."""
    existing_item = collection.find_one({"name": object_label})
    if movement_direction == "added":
        if existing_item:
            # Increase quantity if item exists
            new_quantity = existing_item["quantity"] + count
            collection.update_one(
                {"name": object_label},
                {"$set": {"quantity": new_quantity}},
            )
            print(f"Item added: {object_label}, New Quantity: {new_quantity}")
        else:
            # Insert new item if not found
            collection.insert_one(
                {"name": object_label, "quantity": count, "unit": None}
            )
            print(f"New item added: {object_label}, Quantity: {count}")
    elif movement_direction == "removed":
        if existing_item:
            # Decrease quantity if item exists and quantity is greater than 0
            new_quantity = max(0, existing_item["quantity"] - count)
            collection.update_one(
                {"name": object_label},
                {"$set": {"quantity": new_quantity}},
            )
            print(f"Item removed: {object_label}, New Quantity: {new_quantity}")


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    # Perform object detection
    results = model(frame)
    if not results:
        print("No results returned from model")
        continue

    results = results[0]
    detections = sv.Detections.from_ultralytics(results)

    # Clear previous object counts
    object_counts.clear()

    # Iterate through detected objects
    for i in range(len(detections.class_id)):
        class_id = detections.class_id[i]
        object_label = class_names[class_id]  # Get class name from ID

        # Get bounding box
        box = detections.xyxy[i]
        center_y = (box[1] + box[3]) / 2  # Vertical center of the bounding box

        # Count objects
        if object_label in object_counts:
            object_counts[object_label] += 1
        else:
            object_counts[object_label] = 1

        # Movement tracking logic
        if object_label in object_positions:
            previous_center_y = object_positions[object_label]
            if center_y < previous_center_y:
                movement_direction = "removed"  # Moving from bottom to top (removed)
            else:
                movement_direction = "added"  # Moving from top to bottom (added)
        else:
            movement_direction = "added"  # First detection, assume added

        # Update MongoDB for this object based on its movement
        print(
            f"Object: {object_label}, Movement: {movement_direction}, Count: {object_counts[object_label]}"
        )
        update_mongo_db(object_label, movement_direction, object_counts[object_label])

        # Store the current position for the next frame comparison
        object_positions[object_label] = center_y

    previous_counts = object_counts.copy()

    # Annotate and display the frame with counts
    annotated_image = frame.copy()
    for i in range(len(detections.class_id)):
        class_id = detections.class_id[i]
        object_label = class_names[class_id]
        count = object_counts.get(object_label, 0)

        # Draw bounding box
        box = detections.xyxy[i]
        cv2.rectangle(
            annotated_image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            2,
        )

        # Add label with count
        label = f"{object_label}: {count}"
        cv2.putText(
            annotated_image,
            label,
            (int(box[0]), int(box[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Webcam", annotated_image)

    if cv2.waitKey(1) % 256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()
mongo_client.close()
