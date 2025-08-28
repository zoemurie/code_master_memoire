from flask import Flask, render_template, request, redirect, url_for, send_from_directory 
import os 
import pytesseract 
from PIL import Image 
from werkzeug.utils import secure_filename 
import re 
import cv2 
import numpy as np 

app = Flask(__name__)

# Configuration du dossier d'upload 
UPLOAD_FOLDER = 'uploads' 
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'} 

def correct_skew(image_path): 
    img = cv2.imread(image_path) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # Détection des bords 
    edges = cv2.Canny(gray, 50, 150, apertureSize=3) 

    # Détection des lignes avec Hough Transform 
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200) 
    if lines is not None: 
        angles = [] 
        for rho, theta in lines[:, 0]: 
            angle = (theta - np.pi / 2) * 180 / np.pi # Convertir en degrés 
            angles.append(angle) 
        # Calcul de l'angle médian pour éviter les erreurs 
        skew_angle = np.median(angles) 
        # Rotation inverse de l'image 
        (h, w) = img.shape[:2] 
        center = (w // 2, h // 2) 
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0) 
        img_rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE) 
        # Sauvegarde de l'image corrigée 
        corrected_path = image_path.replace(".jpg", "_corrected.jpg").replace(".png", "_corrected.png").replace(".jpeg", "_corrected.jpeg")
        cv2.imwrite(corrected_path, img_rotated) 
        return corrected_path 
    return image_path # Retourner l'image d'origine si pas d'angle détecté 

def preprocess_image(image_path): 
    # Correction de l'inclinaison 
    corrected_path = correct_skew(image_path) 
    img = cv2.imread(corrected_path, cv2.IMREAD_GRAYSCALE) 
    # Réduction du bruit 
    blurred = cv2.GaussianBlur(img, (5, 5), 0) 
    # Binarisation (seuillage) 
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    # Sauvegarde de l'image traitée
    preprocessed_path = corrected_path.replace(".jpg", "_processed.jpg").replace(".png", "_processed.png").replace(".jpeg", "_processed.jpeg")
    cv2.imwrite(preprocessed_path, thresh) 
    return preprocessed_path 

def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'] 

def extract_info(text): 
    # Normaliser les espaces et caractères parasites 
    text = text.replace('|', ' ').replace('Ne:', ' ').replace('$', ' ') 
    
    # Extraction du numéro de carte (12 chiffres) 
    identity_number_match = re.search(r'\b\d{12}\b', text) 
    if identity_number_match: 
        identity_number = identity_number_match.group(0) 
    else: 
        all_digits = re.findall(r'\d', text) 
        identity_number = ''.join(all_digits)[:12].ljust(12, '?') 
    
    # Extraction de la date de naissance (xx.xx.xxxx) 
    dob_match = re.search(r'(\d{2}\.\d{2}\.\d{4})', text) 
    if dob_match: 
        date_of_birth = dob_match.group(1) 
    else: 
        # Récupérer tous les chiffres du texte 
        all_digits = re.findall(r'\d', text) 
        if len(all_digits) >= 8: 
            date_of_birth = f"{''.join(all_digits[:2])}.{''.join(all_digits[2:4])}.{''.join(all_digits[4:8])}" 
        else: 
            date_of_birth = "??.??.????" 
    
    # Autres informations 
    surname = re.search(r'Nom\s*:\s*(\S+)', text) 
    name = re.search(r'Prénom\(s\)\s*:\s*(.+)', text) 
    sex = re.search(r'Sexe\s*:\s*(\S+)', text) 
    pob = re.search(r'à\s*([A-Za-zÀ-ÿ\s-]+)', text) 
    height = re.search(r'Taille\s*:\s*(\S+)', text) 
    
    extracted_data = { 
        "Identity number": identity_number, 
        "Surname": surname.group(1) if surname else "Not found", 
        "Name(s)": name.group(1) if name else "Not found", 
        "Sex": sex.group(1) if sex else "Not found", 
        "Date of birth": date_of_birth, 
        "Place of birth": pob.group(1) if pob else "Not found", 
        "Height": height.group(1) if height else "Not found" 
    } 
    return extracted_data 

def extract_info_with_confidence(image_path): 
    # Charger l'image prétraitée 
    img = Image.open(image_path) 

    # Utiliser image_to_data pour obtenir des informations sur la confiance de l'OCR 
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT) 
    
    # Calculer la confiance moyenne des mots extraits 
    confidences = [] 
    for conf in ocr_data['conf']: 
        if conf != '-1': # Ignorer les valeurs non définies 
            confidences.append(int(conf)) 
    if confidences: 
        average_confidence = sum(confidences) / len(confidences) 
    else: 
        average_confidence = 0 
        
    # Extraire le texte 
    text = pytesseract.image_to_string(img) 
    
    # Calculer la confiance 
    confidence_threshold = 80 # Par exemple, un seuil de confiance à 80% 
    is_confident = average_confidence >= confidence_threshold 
    
    return text, average_confidence, is_confident 

def detect_document_type(text): 
    if "passeport" in text.lower(): 
        return "Passport" 
    elif "CARTE NATIONALE" in text: 
        if "IDENTITY CARD" in text: 
            return "New ID card model" 
        else: 
            return "Old ID card model" 
    return "Unknown document" 

@app.route('/', methods=['GET', 'POST']) 
def upload_file(): 
    if request.method == 'POST': 
        if 'file' not in request.files: 
            return "No file sent" 
        file = request.files['file'] 
        if file.filename == '': 
            return "No files selected" 
        if file and allowed_file(file.filename): 
            filename = secure_filename(file.filename) 
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
            file.save(file_path) 
            return redirect(url_for('uploaded_file', filename=filename)) 
    return ''' 
    <!doctype html> 
    <title>Upload an image</title> 
    <h1>Upload an image</h1> 
    <form method=post enctype=multipart/form-data> 
    <input type=file name=file accept="image/*"> 
    <input type=submit value=Uploader> 
    </form> 
    ''' 

@app.route('/uploads/<filename>') 
def uploaded_file(filename): 
    return f''' 
    <!doctype html>
    <title>Image uploadée</title>
    <h1>Image uploadée avec succès</h1>
    <img src="{url_for("get_file", filename=filename)}" width="600"> 
    <br><br>
    <form action="{url_for("analyze", filename=filename)}" method="post"> 
    <input type="submit" value="Analyser l'image"> 
    </form> 
    ''' 

@app.route('/uploads/files/<filename>') 
def get_file(filename): 
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename) 

@app.route('/analyze/<filename>', methods=['POST']) 
def analyze(filename): 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
    # Appliquer le prétraitement sur l'image (inclus correction de l'inclinaison, binarisation, etc.) 
    preprocessed_image_path = preprocess_image(image_path) 
    # Extraction du texte et confiance à partir de l'image prétraitée 
    text, average_confidence, is_confident = extract_info_with_confidence(preprocessed_image_path) 
    # Extraire les informations si le niveau de confiance est suffisant 
    extracted_data = extract_info(text) 
    
    document_type = detect_document_type(text) 
    
    return f''' 
    <!doctype html>
    <title>Résultats de l'analyse</title>
    <h1>Image après prétraitement :</h1> 
    <img src="{url_for('get_file', filename=os.path.basename(preprocessed_image_path))}" width="600"> 
    
    <h1>Texte détecté :</h1> 
    <pre>{text}</pre> 
    
    <h2>Niveau de confiance:</h2> 
    <p>Confiance moyenne: {average_confidence:.1f}%</p> 
    <p>Confiance d'extraction: {'Élevée' if is_confident else 'Faible'}</p> 
    
    <h2>Type de document détecté:</h2> 
    <p><strong>{document_type}</strong></p> 
    
    <h2>Informations extraites:</h2> 
    <p><strong>Numéro d'identité:</strong> {extracted_data["Identity number"]}</p> 
    <p><strong>Nom:</strong> {extracted_data['Surname']}</p>
    <p><strong>Prénom(s):</strong> {extracted_data['Name(s)']}</p> 
    <p><strong>Sexe:</strong> {extracted_data['Sex']}</p> 
    <p><strong>Date de naissance:</strong> {extracted_data['Date of birth']}</p> 
    <p><strong>Lieu de naissance:</strong> {extracted_data['Place of birth']}</p> 
    <p><strong>Taille:</strong> {extracted_data['Height']}</p> 
    
    <br>
    <a href="{url_for('upload_file')}">Analyser une nouvelle image</a>
    ''' 

if __name__ == '__main__': 
    app.run(debug=True)