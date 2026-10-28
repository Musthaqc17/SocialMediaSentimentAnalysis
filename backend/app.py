
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract

from backend.predict import predict_sentiment

# Configure Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in {"png", "jpg", "jpeg"}
    )


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Social Media Sentiment Analysis API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a valid JSON object."
        }), 400

    text = data.get("text", "")
    model_name = data.get("model", "SVM")

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "error": "Please provide text for sentiment analysis."
        }), 400

    try:
        result = predict_sentiment(text, model_name)
        return jsonify(result)

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs."
        }), 500


@app.route("/predict-image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({
            "error": "Please upload an image."
        }), 400

    image_file = request.files["image"]
    model_name = request.form.get("model", "SVM")

    if not image_file.filename:
        return jsonify({
            "error": "No image selected."
        }), 400

    if not allowed_file(image_file.filename):
        return jsonify({
            "error": "Only PNG, JPG and JPEG images are supported."
        }), 400

    try:
        with Image.open(image_file.stream) as img:
            image = img.convert("RGB")

            extracted_text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

        if not extracted_text:
            return jsonify({
                "error": "No readable text found in image."
            }), 422

        result = predict_sentiment(
            extracted_text,
            model_name
        )

        return jsonify({
            "extracted_text": extracted_text,
            "sentiment_result": result
        })

    except FileNotFoundError as error:
        return jsonify({
            "error": str(error)
        }), 500

    except (OSError, ValueError) as error:
        return jsonify({
            "error": f"Could not process image: {error}"
        }), 400

    except Exception:
        app.logger.exception("Image analysis failed")
        return jsonify({
            "error": "Image analysis failed. Check server logs."
        }), 500


@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image too large. Maximum size is 10 MB."
    }), 413


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
