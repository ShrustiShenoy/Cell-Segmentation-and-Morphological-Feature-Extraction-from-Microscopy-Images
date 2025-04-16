import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === Paths ===
INPUT_IMAGE = "data/focal9.tif"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load Image ===
def load_image(path):
    full_path = os.path.abspath(path)
    print(f"🔍 Reading image from: {full_path}")
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"❌ Image not found at {full_path}")
    return image

# === Preprocessing ===
def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred

# === Thresholding ===
def threshold_image(image):
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# === Contour Detection ===
def detect_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# === Feature Extraction ===
def extract_features(contours):
    features = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area / (perimeter**2)) if perimeter != 0 else 0
        features.append({
            "Cell_ID": i+1,
            "Area": area,
            "Perimeter": perimeter,
            "Circularity": circularity
        })
    return pd.DataFrame(features)

# === Draw Annotated Contours ===
def draw_annotated(image, contours):
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(annotated, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
    return annotated

# === Main Execution ===
def main():
    image = load_image(INPUT_IMAGE)
    preprocessed = preprocess_image(image)
    thresholded = threshold_image(preprocessed)
    contours = detect_contours(thresholded)
    features_df = extract_features(contours)

    # Save image with annotations
    annotated_image = draw_annotated(image, contours)
    output_img_path = os.path.join(OUTPUT_FOLDER, "annotated_cells.png")
    cv2.imwrite(output_img_path, annotated_image)

    # Save CSV
    output_csv_path = os.path.join(OUTPUT_FOLDER, "cell_features.csv")
    features_df.to_csv(output_csv_path, index=False)

    print(f"✅ Annotated image saved to: {output_img_path}")
    print(f"📊 Cell features saved to: {output_csv_path}")
    print(f"🔢 Total cells detected: {len(contours)}")
    print("📈 Summary Stats:")
    print(features_df.describe())

    # Optional Visualization
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Cells: {len(contours)}")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
