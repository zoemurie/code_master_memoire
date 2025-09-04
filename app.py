from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import time

from config import Config
from ocr_processor import DocumentProcessor
from document_analyzer import analyze_document

app = Flask(__name__)
app.config.from_object(Config)

processor = DocumentProcessor()

@app.route('/')
def index():
    """Page d'accueil"""
    system_info = processor.get_system_info()
    return render_template('index.html', system=system_info)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Gestion upload"""
    if 'file' not in request.files:
        return render_template('index.html', error="Aucun fichier envoyé")
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return render_template('index.html', error="Fichier invalide")
    
    # Sauvegarde sécurisée
    filename = secure_filename(f"{int(time.time())}_{file.filename}")
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    return redirect(url_for('analyze_document_route', filename=filename))

@app.route('/analyze/<filename>')
def analyze_document_route(filename):
    """Analyse du document"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return render_template('error.html', error="Fichier introuvable"), 404
    
    # Analyse
    analysis_result = analyze_document(file_path, processor)
    
    return render_template('results.html', 
                         result=analysis_result, 
                         filename=filename)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API JSON pour analyse"""
    # Endpoint API séparé
    pass

def allowed_file(filename):
    """Vérification extension fichier"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS'])

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, port=8080)