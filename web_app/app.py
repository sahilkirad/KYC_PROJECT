import os
# Set environment variable to resolve OpenMP conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from flask import Flask, request, render_template
import sys
import logging

# Add the parent directory to sys.path to import extract_text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.processing.extract_text_enhanced import extract_text_from_image

app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename='D:/KYC_Project/logs/web_app_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure upload folder
UPLOAD_FOLDER = 'D:/KYC_Project/web_app/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    image_path = None
    cropped_image_path = None
    error = None

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            error = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No selected file'
            elif not allowed_file(file.filename):
                error = 'Invalid file type. Please upload a .jpg, .jpeg, or .png file.'
            else:
                # Save the uploaded file
                filename = file.filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                logging.info(f"Uploaded file saved to {filepath}")

                # Process the image
                try:
                    model_path = 'D:/KYC_Project/models/detection/kyc_model2/weights/best.pt'
                    result = extract_text_from_image(filepath, model_path)
                    logging.info(f"Processing result: {result}")

                    if result:
                        # Prepare paths for display
                        image_path = f"static/uploads/{filename}"
                        # Check if cropped image exists
                        cropped_filename = f"{result['class']}_{filename}"
                        cropped_full_path = os.path.join('D:/KYC_Project/outputs/debug', cropped_filename)
                        if os.path.exists(cropped_full_path):
                            # Copy cropped image to static/uploads for display
                            cropped_static_path = os.path.join(app.config['UPLOAD_FOLDER'], f"cropped_{cropped_filename}")
                            os.replace(cropped_full_path, cropped_static_path)
                            cropped_image_path = f"static/uploads/cropped_{cropped_filename}"
                        else:
                            error = 'Cropped image not found for debugging.'
                    else:
                        error = 'Failed to process the image. Check logs for details.'
                except Exception as e:
                    error = f'Error processing image: {str(e)}'
                    logging.error(f"Error in processing: {str(e)}")

    return render_template('index.html', result=result, image_path=image_path, 
                         cropped_image_path=cropped_image_path, error=error)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)