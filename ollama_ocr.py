import os
import logging
import base64
from flask import Flask, request, render_template, jsonify
from groq import Groq
from paddleocr import PaddleOCR
import csv
import mimetypes

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Initialize OCR and Groq
try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)
    logging.info("PaddleOCR initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize PaddleOCR: {str(e)}")
    raise

try:
    client = Groq(api_key="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")
    logging.info("Groq client initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {str(e)}")
    raise

# Global variables
image_path = None
csv_path = 'data.csv'
log_csv_path = 'refined_text_log.csv'

# Original logic functions
def log_refined_text(refined_text):
    """
    Logs the refined OCR-extracted text into a CSV file.
    """
    try:
        with open(log_csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([refined_text])
        logging.debug(f"Logged refined text to {log_csv_path}")
    except Exception as e:
        logging.error(f"Error logging refined text: {str(e)}")
        raise

def extract_and_clean_text(image_path):
    """
    Extracts text from an image using OCR and cleans it.
    """
    try:
        logging.debug(f"Processing image with OCR: {image_path}")
        results = ocr.ocr(image_path, cls=True)
        if not results or not results[0]:
            logging.warning(f"No text detected in image: {image_path}")
            raise ValueError("No readable text detected in the image")
        logging.debug(f"OCR results: {results}")
        cleaned_text = []
        for line in results[0]:
            text = line[1][0]
            if len(text) >= 3 and text.isprintable():
                cleaned_text.append(text)
        logging.debug(f"Cleaned text: {cleaned_text}")
        return cleaned_text
    except Exception as e:
        logging.error(f"Error in extract_and_clean_text: {str(e)}")
        raise

def process_image_and_csv(image_path, csv_path):
    """
    Processes a nutritional image, sending data to the Groq API for analysis.
    """
    if not image_path:
        logging.error("No image path provided")
        return "Error: No image path provided!", "", ""
    try:
        # Extract text from the image
        cleaned_text = extract_and_clean_text(image_path)
        extracted_text = ", ".join(cleaned_text)
        logging.debug(f"Extracted text with numbers: {extracted_text}")

        # Determine MIME type from file extension
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Fallback to JPEG if unknown
        logging.debug(f"Detected MIME type: {mime_type}")

        # Read the image file and encode it as base64
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            logging.error(f"Failed to read image file: {str(e)}")
            raise ValueError(f"Cannot read image file: {str(e)}")
        image_data_url = f"data:{mime_type};base64,{base64_image}"
        logging.debug("Image encoded as base64 for Groq")

        # Sending the OCR text to NLP model with base64 image data
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I am using OCR to extract the Nutritional information from the Food pack labels. The extracted text is: {extracted_text}. Refine this text and return only the nutritional facts, ingredients, and food name in the following format:\n\nNutritional Facts:\n[Fact 1]\n[Fact 2]\n...\n\nIngredients:\n[Ingredient 1], [Ingredient 2], ...\n\nFood Name: [Name]\n\nDo not include any explanations, steps, or additional text beyond this format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url  # Use base64-encoded image
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None
        )
        logging.debug(f"Groq response: {response}")
        refined_text = response.choices[0].message.content
        log_refined_text(refined_text)
        return refined_text, "", ""
    except Exception as e:
        logging.error(f"Error in process_image_and_csv: {str(e)}")
        return f"Error occurred: {str(e)}", "", ""

# Flask setup
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB file size limit

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_nutritional', methods=['POST'])
def upload_nutritional():
    global image_path
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logging.debug(f"Saving file to: {filepath}")
        file.save(filepath)
        if not os.path.exists(filepath):
            logging.error("File save failed")
            return jsonify({"error": "Failed to save the uploaded file"}), 500
        image_path = f"/{filepath}"  # Relative URL for frontend access
        refined_text, _, _ = process_image_and_csv(filepath, csv_path)  # Pass local path for processing
        if refined_text.startswith("Error occurred:"):
            logging.error(f"Processing failed: {refined_text}")
            return jsonify({"error": refined_text}), 500
        logging.info("Nutritional image processed successfully")
        return jsonify({"refined_text": refined_text, "image_url": image_path})
    except Exception as e:
        logging.error(f"Upload nutritional error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/upload_medical', methods=['POST'])
def upload_medical():
    if 'file' not in request.files:
        logging.error("No file part in request")
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({"error": "No file selected"}), 400
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        logging.debug(f"Saving file to: {filepath}")
        file.save(filepath)
        if not os.path.exists(filepath):
            logging.error("File save failed")
            return jsonify({"error": "Failed to save the uploaded file"}), 500
        
        # Extract text from the image
        cleaned_text = extract_and_clean_text(filepath)
        extracted_text = ", ".join(cleaned_text)
        logging.debug(f"Extracted medical text: {extracted_text}")

        # Determine MIME type from file extension
        mime_type, _ = mimetypes.guess_type(filepath)
        if not mime_type or not mime_type.startswith('image/'):
            mime_type = 'image/jpeg'  # Fallback to JPEG if unknown
        logging.debug(f"Detected MIME type: {mime_type}")

        # Read the image file and encode it as base64
        try:
            with open(filepath, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except IOError as e:
            logging.error(f"Failed to read image file: {str(e)}")
            raise ValueError(f"Cannot read image file: {str(e)}")
        image_data_url = f"data:{mime_type};base64,{base64_image}"
        logging.debug("Image encoded as base64 for Groq")

        # Sending the OCR text to NLP model with base64 image data
        response = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"I am using OCR to extract text from a medical report. The extracted text is: {extracted_text}. Refine this text and return only the important medical details and diagnosis (if available) in the following format:\n\nMedical Details:\n[Detail 1]\n[Detail 2]\n...\n\nDiagnosis: [Diagnosis]\n\nIf no diagnosis is present, omit the Diagnosis section. Do not include any explanations, steps, or additional text beyond this format."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url  # Use base64-encoded image
                            }
                        }
                    ]
                }
            ],
            temperature=0.7,  # Matching original medical route
            max_tokens=300,  # Matching original medical route
            top_p=1,
            stream=False,
            stop=None
        )
        logging.debug(f"Groq response: {response}")
        refined_text = response.choices[0].message.content
        logging.info("Medical report processed successfully")
        return jsonify({"refined_text": refined_text})
    except Exception as e:
        logging.error(f"Upload medical error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/evaluate_combined', methods=['POST'])
def evaluate_combined():
    data = request.json
    nutritional_text = data.get('nutritional_text', '').strip()
    medical_text = data.get('medical_text', '').strip()
    selected_model = data.get('model', 'llama-3.3-70b-versatile')
    selected_language = data.get('language', 'English')

    if not nutritional_text:
        logging.error("No nutritional text provided")
        return jsonify({"error": "Please analyze nutritional data first"}), 400
    if not medical_text:
        logging.error("No medical text provided")
        return jsonify({"error": "Please process medical report first"}), 400

    try:
        next_prompt = f"""
        Dear User,

        Based on the extracted text from your food pack labels: {nutritional_text},
        and the details from your medical report: {medical_text},
        please evaluate the ingredients for safety.
        Provide a short recommendation on whether the food is safe to consume,
        including the safe quantity for intake if applicable.
        If the food is not recommended, briefly explain why it should be avoided.

        Please provide the response in the following format:

        1. First, a short and clear recommendation in **English**.
        2. After that, a short and clear recommendation in **{selected_language}** that corresponds to the English response.
        """
        final_response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "system", "content": "You are a professional medical advisor."},
                      {"role": "user", "content": next_prompt}],
            temperature=0.7,
            max_tokens=400,
            top_p=1,
            stream=False
        )
        logging.info("Combined evaluation completed successfully")
        return jsonify({"result": final_response.choices[0].message.content})
    except Exception as e:
        logging.error(f"Evaluate combined error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

#if __name__ == "__main__":
#    try:
#        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#        logging.info(f"Upload folder created/verified: {UPLOAD_FOLDER}")
#        app.run(debug=True, host='0.0.0.0', port=5000)
#    except Exception as e:
#        logging.error(f"Startup error: {str(e)}")