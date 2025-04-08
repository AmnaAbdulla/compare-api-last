from flask import Flask
from app import app as classify_app
from food_detection_api import app as compare_app

main_app = Flask(__name__)

# Register routes from classify app under "/classify_api"
@main_app.route("/classify", methods=["POST"])
def classify_route():
    return classify_app.view_functions['classify']()

# Register routes from compare app under "/compare_api"
@main_app.route("/compare", methods=["POST"])
def compare_route():
    return compare_app.view_functions['classify_images']()

if __name__ == "__main__":
    import os
    from waitress import serve
    port = int(os.environ.get("PORT", 8080))
    serve(main_app, host="0.0.0.0", port=port)
