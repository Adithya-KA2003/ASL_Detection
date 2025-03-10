import os

dataset_path = "C:/Users/adith/OneDrive/Desktop/ASL/Data"  # Update with your dataset path
class_counts = {}

for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_folder):  # Ensure it's a directory
        class_counts[class_name] = len(os.listdir(class_folder))

# Print dataset distribution
for class_name, count in class_counts.items():
    print(f"Class {class_name}: {count} images")


if __name__ == "__main__":
    print("🚀 Flask is running on http://127.0.0.1:5000/")
    app.run(debug=True, host="127.0.0.1", port=5000)
