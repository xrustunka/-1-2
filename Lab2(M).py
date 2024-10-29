import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Завантажуємо OpenCV
install("opencv-python")

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# Глобальні змінні для зберігання мінімальних та максимальних векторів
min_vectors = {}
max_vectors = {}

# Функції для обробки зображень
def crop_image(image):
    coordinates = np.column_stack(np.where(image == 0))
    if coordinates.size == 0:
        return image

    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)

    cropped_image = image[top_left[0]:bottom_right[0] + 1, top_left[1]:bottom_right[1] + 1]
    return cropped_image

def process_image(file_path, threshold_value):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    _, thresh_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    cropped_image = crop_image(thresh_image)
    return cropped_image

def calculate_feature_vector(image, num_cells):
    height, width = image.shape
    feature_vector = []

    cell_height = height // num_cells
    cell_width = width // num_cells

    for row in range(num_cells):
        for col in range(num_cells):
            cell = image[row * cell_height:(row + 1) * cell_height, col * cell_width:(col + 1) * cell_width]
            black_pixels = np.sum(cell == 0)
            feature_vector.append(black_pixels)

    return feature_vector

def display_image(image, num_cells, name_image):
    height, width = image.shape
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')

    cell_height = height // num_cells
    cell_width = width // num_cells

    for row in range(num_cells):
        for col in range(num_cells):
            rect = plt.Rectangle((col * cell_width, row * cell_height), cell_width, cell_height, 
                                 edgecolor='r', facecolor='none', linewidth=1)
            ax.add_patch(rect)

    plt.title(name_image)
    plt.show()

def normalize_sum(vector):
    total_sum = sum(vector)
    if total_sum == 0:
        return [0 for v in vector]
    return [v / total_sum for v in vector]

def normalize_max(vector):
    max_value = max(vector)
    if max_value == 0:
        return [0 for v in vector]
    return [v / max_value for v in vector]

def print_feature_info(image_file, feature_vector, normalized_sum_vector, normalized_max_vector):
    print("-" * 50)
    print(f"Image: {image_file}")
    print(f"absolute vector: {[int(val) for val in feature_vector]}")
    print(f"Normalized vector (by sum): {[float(val) for val in normalized_sum_vector]}")
    print(f"Normalized vector (by max): {[float(val) for val in normalized_max_vector]}")

def classify_unknown_image(unknown_vector, min_vector, max_vector):
    for i in range(len(min_vector)):
        if not (min_vector[i] <= unknown_vector[i] <= max_vector[i]):
            return False
    return True

def classify_image(normalized_vector_sum):
    class_names = ["Class_1", "Class_2", "Class_3"]
    for class_num in range(1, 4):
        if classify_unknown_image(normalized_vector_sum, min_vectors[class_num], max_vectors[class_num]):
            return class_names[class_num - 1]
    return None

def process_image_file():
    image_path = file_entry.get()
    if not os.path.isfile(image_path):
        result_label.config(text="Неправильний шлях до зображення!")
        return

    try:
        threshold = int(threshold_entry.get())
        num_cells = int(cells_entry.get())
    except ValueError:
        result_label.config(text="Будь ласка, введіть дійсні числа для порогового значення і кількості комірок!")
        return

    image = process_image(image_path, threshold)
    feature_vector = calculate_feature_vector(image, num_cells)

    normalized_vector_sum = normalize_sum(feature_vector)
    normalized_vector_max = normalize_max(feature_vector)

    print_feature_info(os.path.basename(image_path), feature_vector, normalized_vector_sum, normalized_vector_max)
    display_image(image, num_cells, os.path.basename(image_path))

def classify_uploaded_image():
    image_path = file_entry.get()
    if not os.path.isfile(image_path):
        result_label.config(text="Неправильний шлях до зображення!")
        return

    try:
        threshold = int(threshold_entry.get())
        num_cells = int(cells_entry.get())
    except ValueError:
        result_label.config(text="Будь ласка, введіть дійсні числа для порогового значення і кількості комірок!")
        return

    image = process_image(image_path, threshold)
    feature_vector = calculate_feature_vector(image, num_cells)
    normalized_vector_sum = normalize_sum(feature_vector)

    result = classify_image(normalized_vector_sum)
    if result:
        result_label.config(text=f"Unknown image belongs to: {result}")
    else:
        result_label.config(text="Unknown image does not belong to any class.")

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

def main():
    global min_vectors, max_vectors

    # Введення еталонних образів
    class1_images = [
        "D:/image/class1/1.png",
        "D:/image/class1/2.png",
        "D:/image/class1/3.png",
        "D:/image/class1/4.png",
        "D:/image/class1/5.png",
        "D:/image/class1/6.png"
    ]

    class2_images = [
        "D:/image/class2/1.png",
        "D:/image/class2/2.png",
        "D:/image/class2/3.png",
        "D:/image/class2/4.png",
        "D:/image/class2/5.png",
        "D:/image/class2/6.png"
    ]

    class3_images = [
        "D:/image/class3/1.png",
        "D:/image/class3/2.png",
        "D:/image/class3/3.png",
        "D:/image/class3/4.png",
        "D:/image/class3/5.png",
        "D:/image/class3/6.png"
    ]

    threshold = 150
    num_cells = 5

    # Обробка еталонних образів
    class_vectors = {1: [], 2: [], 3: []}
    for class_num, images in zip(range(1, 4), [class1_images, class2_images, class3_images]):
        for image_file in images:
            image = process_image(image_file, threshold)
            feature_vector = calculate_feature_vector(image, num_cells)
            normalized_vector_sum = normalize_sum(feature_vector)
            normalized_vector_max = normalize_max(feature_vector)

            class_vectors[class_num].append((feature_vector, normalized_vector_sum, normalized_vector_max))

    # Обчислення мінімальних та максимальних векторів
    min_vectors = {1: [], 2: [], 3: []}
    max_vectors = {1: [], 2: [], 3: []}
    for class_num in range(1, 4):
        min_vectors[class_num] = [min(vec[1][i] for vec in class_vectors[class_num]) for i in range(num_cells)]
        max_vectors[class_num] = [max(vec[1][i] for vec in class_vectors[class_num]) for i in range(num_cells)]

    root = tk.Tk()
    root.title("Image Classifier")
    root.geometry("500x400")
    root.configure(bg="#f0f0f0")

    # Вибір зображення
    tk.Label(root, text="Виберіть зображення:", bg="#f0f0f0").pack(pady=5)
    global file_entry
    file_entry = tk.Entry(root, width=50)
    file_entry.pack(pady=5)
    tk.Button(root, text="Browse", command=browse_file).pack(pady=5)

    # Порогове значення
    tk.Label(root, text="Порогове значення:", bg="#f0f0f0").pack(pady=5)
    global threshold_entry
    threshold_entry = tk.Entry(root, width=10)
    threshold_entry.pack(pady=5)
    threshold_entry.insert(0, "150")

    # Кількість комірок
    tk.Label(root, text="Кількість комірок:", bg="#f0f0f0").pack(pady=5)
    global cells_entry
    cells_entry = tk.Entry(root, width=10)
    cells_entry.pack(pady=5)
    cells_entry.insert(0, "5")

    # Кнопка обробки зображення
    tk.Button(root, text="Process Image", command=process_image_file).pack(pady=10)
    
    # Кнопка класифікації зображення
    tk.Button(root, text="Classify Image", command=classify_uploaded_image).pack(pady=10)

    # Результат
    global result_label
    result_label = tk.Label(root, text="", bg="#f0f0f0")
    result_label.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
