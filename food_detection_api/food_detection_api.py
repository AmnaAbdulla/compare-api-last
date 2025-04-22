from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import traceback
import os

app = Flask(__name__)

def url_to_image(url):
    """Download an image from a URL and convert it to OpenCV format."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to download image: {url}")
            return None

        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            print(f"Failed to decode image: {url}")
            return None

        return image

    except Exception as e:
        print(f"Error downloading image {url}: {str(e)}")
        return None

def classify_food_image(image_url, is_second_image=False):
    """Process image and detect if it qualifies as food on a plate or an empty/eaten plate."""
    image = url_to_image(image_url)
    if image is None:
        print(f"Image could not be loaded: {image_url}")
        return False

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
                                   param1=100, param2=30, minRadius=50, maxRadius=400)

        if circles is None:
            print("No plate detected.")
            return False

        circles = np.uint16(np.around(circles))
        (x, y, r) = circles[0][0]

        plate_mask = np.zeros_like(gray)
        cv2.circle(plate_mask, (x, y), r, 255, -1)
        plate_area = cv2.bitwise_and(image, image, mask=plate_mask)

        hsv = cv2.cvtColor(plate_area, cv2.COLOR_BGR2HSV)

        plate_pixel_count = np.sum(plate_mask > 0)

        # Detect curry stains (yellow-orange)
        lower_curry = np.array([10, 100, 100])
        upper_curry = np.array([25, 255, 255])
        curry_mask = cv2.inRange(hsv, lower_curry, upper_curry)
        curry_coverage = np.sum(curry_mask > 0) / plate_pixel_count

        # Detect any colored stains (exclude low saturation = grayscale)
        lower_color = np.array([0, 50, 50])
        upper_color = np.array([179, 255, 255])
        color_mask = cv2.inRange(hsv, lower_color, upper_color)
        color_coverage = np.sum(color_mask > 0) / plate_pixel_count

        # Detect mostly white plate
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 40, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        white_coverage = np.sum(white_mask > 0) / plate_pixel_count

        # Detect colored plates (non-white dominant surfaces)
        colored_plate = white_coverage < 0.6

        # Original food detection
        lower_food = np.array([5, 50, 50])
        upper_food = np.array([35, 255, 255])
        food_mask = cv2.inRange(hsv, lower_food, upper_food)
        food_coverage = np.sum(food_mask > 0) / plate_pixel_count

        # Additional yellow stain detection
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_coverage = np.sum(yellow_mask > 0) / plate_pixel_count

        if is_second_image:
            if curry_coverage > 0.05:
                print(f"✅ Curry stains detected: {image_url}")
                return True
            elif color_coverage > 0.05:
                print(f"✅ Colored stains detected: {image_url}")
                return True
            elif white_coverage > 0.8:
                print(f"✅ Plain white plate detected: {image_url}")
                return True
            elif colored_plate:
                print(f"✅ Colored plate detected: {image_url}")
                return True
            else:
                print(f"❌ Second image rejected (no stains, food remains, or color clue): {image_url}")
                return False

        else:
            if yellow_coverage > 0.5:
                print(f"❌ First image rejected (too much yellow): {image_url}")
                return False

            if food_coverage > 0.2:
                print(f"✅ Food plate detected: {image_url}")
                return True
            else:
                print(f"❌ First image rejected (not enough food): {image_url}")
                return False

    except Exception as e:
        print(f"Error processing image {image_url}: {str(e)}")
        return False

@app.route('/compare', methods=['POST'])
def classify_images():
    """API Endpoint: Receives two image URLs and classifies them."""
    try:
        data = request.get_json()
        if not data or "image_url1" not in data or "image_url2" not in data:
            return jsonify({"error": "Invalid input. Both image URLs are required."}), 400

        image_url1 = data["image_url1"]
        image_url2 = data["image_url2"]

        if image_url1 == image_url2:
            print("❌ Same image submitted for both inputs. Rejecting.")
            return jsonify({
                "final_result": False
            })

        first_result = classify_food_image(image_url1)  # Check if it's a food plate
        second_result = classify_food_image(image_url2, is_second_image=True)  # Check if it's an eaten/empty plate

        final_result = first_result and second_result  # True only if both pass

        return jsonify({
            "final_result": bool(final_result),
        })

    except Exception as e:
        error_message = traceback.format_exc()
        print("Error:", error_message)
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Use Render's assigned port
    from waitress import serve
    serve(app, host="0.0.0.0", port=port)
