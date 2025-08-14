from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}

class DocumentProcessorSimulated:
    """
    Processeur de documents simulé qui démontre l'architecture
    sans nécessiter PyTorch pour le développement initial
    """
    def __init__(self):
        self.document_classes = ['CNI', 'Passeport', 'Permis de conduire', 'Autre']
        logger.info("Processeur de documents initialisé (mode simulation)")
        
    def classify_document(self, image_path):
        """Classification simulée basée sur l'analyse d'image avec OpenCV"""
        try:
            # Charger l'image
            image = cv2.imread(image_path)
            if image is None:
                return {'type': 'Erreur', 'confidence': 0.0}
                
            height, width = image.shape[:2]
            
            # Analyse simple basée sur les dimensions et couleurs
            aspect_ratio = width / height
            
            # Simuler la classification basée sur des heuristiques
            if 1.5 < aspect_ratio < 1.7:  # Format carte
                if self._detect_blue_background(image):
                    doc_type = 'CNI'
                    confidence = 0.92
                else:
                    doc_type = 'Permis de conduire'
                    confidence = 0.87
            elif aspect_ratio > 1.3:  # Format passeport
                doc_type = 'Passeport'
                confidence = 0.89
            else:
                doc_type = 'Autre'
                confidence = 0.65
                
            # Simuler les probabilités pour tous les types
            all_probabilities = {}
            for i, class_name in enumerate(self.document_classes):
                if class_name == doc_type:
                    all_probabilities[class_name] = confidence
                else:
                    remaining_prob = (1 - confidence) / (len(self.document_classes) - 1)
                    all_probabilities[class_name] = remaining_prob
                    
            return {
                'type': doc_type,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la classification: {e}")
            return {'type': 'Erreur', 'confidence': 0.0, 'all_probabilities': {}}
    
    def _detect_blue_background(self, image):
        """Détecte si l'image a un fond bleu (caractéristique CNI française)"""
        # Convertir en HSV pour détecter les blues
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Définir la gamme de bleu
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Créer un masque pour le bleu
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Calculer le pourcentage de pixels bleus
        blue_percentage = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        
        return blue_percentage > 0.1  # Si plus de 10% de l'image est bleue
    
    def detect_text_regions(self, image_path):
        """Détection de régions de texte avec OpenCV"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Prétraitement pour améliorer la détection
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Utiliser MSER (Maximally Stable Extremal Regions) pour détecter le texte
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(thresh)
            
            # Convertir les régions en boîtes englobantes
            boxes = []
            for region in regions:
                if len(region) > 50:  # Filtrer les petites régions
                    x, y, w, h = cv2.boundingRect(region)
                    if w > 20 and h > 10:  # Filtrer selon la taille
                        boxes.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
            
            # Créer un masque de visualisation
            mask = np.zeros(gray.shape, dtype=np.uint8)
            for box in boxes:
                cv2.rectangle(mask, (box['x'], box['y']), 
                             (box['x'] + box['width'], box['y'] + box['height']), 255, -1)
            
            # Sauvegarder le masque
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            mask_filename = f"{base_name}_text_mask.jpg"
            mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
            cv2.imwrite(mask_path, mask)
            
            return {
                'mask_path': mask_filename,
                'text_coverage': float(np.mean(mask > 0)),
                'detected_regions': boxes[:10]  # Limiter à 10 régions pour l'affichage
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection: {e}")
            return {'error': str(e), 'detected_regions': [], 'mask_path': None, 'text_coverage': 0.0}
    
    def recognize_text_advanced(self, image_path):
        """Reconnaissance de texte avancée simulée"""
        try:
            # Simuler l'analyse d'image pour extraction d'informations
            image = cv2.imread(image_path)
            
            # Simuler différents résultats selon le type d'image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
            
            # Générer des données simulées mais cohérentes
            if mean_intensity > 128:  # Image claire
                confidence = 0.92
                recognized_text = {
                    'nom': 'MARTIN',
                    'prenom': 'Jean Pierre',
                    'date_naissance': '15.03.1985',
                    'lieu_naissance': 'PARIS',
                    'numero_document': '123456789012',
                    'sexe': 'M',
                    'taille': '175cm'
                }
            else:  # Image sombre, qualité réduite
                confidence = 0.76
                recognized_text = {
                    'nom': 'DUBOIS',
                    'prenom': 'Marie Claire',
                    'date_naissance': '22.07.1992',
                    'lieu_naissance': 'LYON',
                    'numero_document': '987654321098',
                    'sexe': 'F',
                    'taille': '165cm'
                }
            
            return {
                'recognized_text': recognized_text,
                'confidence': confidence,
                'method': 'CNN+RNN (Simulé)',
                'processing_time': 0.15
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la reconnaissance: {e}")
            return {'error': str(e), 'recognized_text': {}}
    
    def extract_structured_information(self, classification_result, text_result):
        """Extrait les informations structurées selon le type de document"""
        doc_type = classification_result.get('type', 'Autre')
        text_data = text_result.get('recognized_text', {})
        
        if doc_type == 'CNI':
            return self._extract_cni_info(text_data)
        elif doc_type == 'Passeport':
            return self._extract_passport_info(text_data)
        elif doc_type == 'Permis de conduire':
            return self._extract_license_info(text_data)
        else:
            return self._extract_generic_info(text_data)
    
    def _extract_cni_info(self, text_data):
        return {
            'document_type': 'Carte Nationale d\'Identité',
            'numero': text_data.get('numero_document', 'Non détecté'),
            'nom': text_data.get('nom', 'Non détecté'),
            'prenom': text_data.get('prenom', 'Non détecté'),
            'date_naissance': text_data.get('date_naissance', 'Non détecté'),
            'lieu_naissance': text_data.get('lieu_naissance', 'Non détecté'),
            'sexe': text_data.get('sexe', 'Non détecté'),
            'taille': text_data.get('taille', 'Non détecté')
        }
    
    def _extract_passport_info(self, text_data):
        return {
            'document_type': 'Passeport',
            'numero_passeport': text_data.get('numero_document', 'Non détecté'),
            'nom': text_data.get('nom', 'Non détecté'),
            'prenom': text_data.get('prenom', 'Non détecté'),
            'date_naissance': text_data.get('date_naissance', 'Non détecté'),
            'lieu_naissance': text_data.get('lieu_naissance', 'Non détecté'),
            'sexe': text_data.get('sexe', 'Non détecté')
        }
    
    def _extract_license_info(self, text_data):
        return {
            'document_type': 'Permis de Conduire',
            'numero_permis': text_data.get('numero_document', 'Non détecté'),
            'nom': text_data.get('nom', 'Non détecté'),
            'prenom': text_data.get('prenom', 'Non détecté'),
            'date_naissance': text_data.get('date_naissance', 'Non détecté'),
            'lieu_naissance': text_data.get('lieu_naissance', 'Non détecté')
        }
    
    def _extract_generic_info(self, text_data):
        return {
            'document_type': 'Document non identifié',
            'informations_detectees': text_data
        }

# Instance globale du processeur
processor = DocumentProcessorSimulated()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Système OCR Avancé - IA pour Documents d'Identité</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; margin-bottom: 30px; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
            .upload-area:hover { border-color: #999; }
            input[type="file"] { margin: 10px; }
            input[type="submit"] { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            input[type="submit"]:hover { background-color: #0056b3; }
            .features { margin-top: 30px; }
            .feature { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .demo-note { background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Système OCR Avancé avec IA</h1>
            <p style="text-align: center; color: #666;">
                Reconnaissance automatique de documents d'identité utilisant Vision par Ordinateur
            </p>
            
            <div class="demo-note">
                <h3>Mode Démonstration</h3>
                <p>Cette version utilise OpenCV et des algorithmes de vision par ordinateur avancés pour simuler les capacités des modèles CNN et RNN. Parfait pour démontrer l'architecture sans les dépendances PyTorch.</p>
            </div>
            
            <form method="post" enctype="multipart/form-data" action="/upload">
                <div class="upload-area">
                    <h3>Télécharger un document d'identité</h3>
                    <input type="file" name="file" accept=".png,.jpg,.jpeg" required>
                    <br><br>
                    <input type="submit" value="Analyser avec l'IA">
                </div>
            </form>
            
            <div class="features">
                <h3> Fonctionnalités implémentées :</h3>
                <div class="feature">
                    <strong> Classification automatique :</strong> Analyse des dimensions et couleurs pour identifier CNI, Passeport, Permis
                </div>
                <div class="feature">
                    <strong> Détection MSER :</strong> Détection de régions de texte avec algorithme MSER (OpenCV)
                </div>
                <div class="feature">
                    <strong> Extraction intelligente :</strong> Simulation de reconnaissance avancée avec heuristiques
                </div>
                <div class="feature">
                    <strong> Structure adaptative :</strong> Parsing selon le type de document détecté
                </div>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "Aucun fichier envoyé", 400
    
    file = request.files['file']
    if file.filename == '':
        return "Aucun fichier sélectionné", 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        return redirect(url_for('analyze_document', filename=filename))
    
    return "Type de fichier non autorisé", 400

@app.route('/analyze/<filename>')
def analyze_document(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return "Fichier non trouvé", 404
    
    # Début du chronométrage
    start_time = time.time()
    
    # Étape 1: Classification du document
    logger.info("Classification du document en cours...")
    classification_result = processor.classify_document(image_path)
    
    # Étape 2: Détection des régions de texte
    logger.info("Détection des régions de texte...")
    detection_result = processor.detect_text_regions(image_path)
    
    # Étape 3: Reconnaissance de texte
    logger.info("Reconnaissance de texte avancée...")
    text_result = processor.recognize_text_advanced(image_path)
    
    # Étape 4: Extraction d'informations structurées
    logger.info("Extraction d'informations structurées...")
    structured_info = processor.extract_structured_information(classification_result, text_result)
    
    # Calcul du temps de traitement
    processing_time = time.time() - start_time
    
    # Génération du rapport
    analysis_report = {
        'filename': filename,
        'processing_time': round(processing_time, 2),
        'timestamp': datetime.now().isoformat(),
        'classification': classification_result,
        'text_detection': detection_result,
        'text_recognition': text_result,
        'structured_information': structured_info
    }
    
    return render_template_string(get_results_template(), 
                                report=analysis_report, 
                                filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint pour l'intégration avec d'autres systèmes"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"api_{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Traitement IA
            start_time = time.time()
            
            classification_result = processor.classify_document(file_path)
            detection_result = processor.detect_text_regions(file_path)
            text_result = processor.recognize_text_advanced(file_path)
            structured_info = processor.extract_structured_information(classification_result, text_result)
            
            processing_time = time.time() - start_time
            
            # Nettoyage du fichier temporaire
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'processing_time': round(processing_time, 2),
                'classification': classification_result,
                'structured_information': structured_info,
                'confidence_score': classification_result.get('confidence', 0)
            })
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement API: {e}")
            return jsonify({'error': f'Erreur de traitement: {str(e)}'}), 500
    
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

def get_results_template():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Résultats de l'Analyse IA</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .success { background-color: #d4edda; border-color: #c3e6cb; }
            .info { background-color: #d1ecf1; border-color: #bee5eb; }
            .warning { background-color: #fff3cd; border-color: #ffeaa7; }
            .demo { background-color: #e7f3ff; border-color: #b3d9ff; }
            .image-preview { max-width: 250px; border-radius: 8px; margin: 10px; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
            .confidence-high { color: #28a745; font-weight: bold; }
            .confidence-medium { color: #ffc107; font-weight: bold; }
            .confidence-low { color: #dc3545; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; font-weight: bold; }
            .back-button { background-color: #6c757d; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-top: 20px; }
            .json-view { background: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 12px; max-height: 300px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> Résultats de l'Analyse IA</h1>
                <p>Traitement terminé en <strong>{{ report.processing_time }}s</strong> | {{ report.timestamp }}</p>
            </div>
            
            <div class="section demo">
                <h2> Mode Démonstration Actif</h2>
                <p>Cette analyse utilise des algorithmes OpenCV avancés pour simuler les capacités des modèles CNN et RNN, sans nécessiter PyTorch. Les résultats démontrent l'architecture et les capacités du système.</p>
            </div>
            
            <!-- Image originale -->
            <div class="section">
                <h2> Document analysé</h2>
                <img src="{{ url_for('uploaded_file', filename=filename) }}" class="image-preview" alt="Document">
            </div>
            
            <!-- Métriques de performance -->
            <div class="metrics">
                <div class="metric">
                    <h3>⚡ Temps de traitement</h3>
                    <p><strong>{{ report.processing_time }}s</strong></p>
                </div>
                <div class="metric">
                    <h3> Type détecté</h3>
                    <p><strong>{{ report.classification.type }}</strong></p>
                </div>
                <div class="metric">
                    <h3> Confiance</h3>
                    <p class="{% if report.classification.confidence > 0.8 %}confidence-high{% elif report.classification.confidence > 0.5 %}confidence-medium{% else %}confidence-low{% endif %}">
                        {{ "%.1f"|format(report.classification.confidence * 100) }}%
                    </p>
                </div>
                <div class="metric">
                    <h3> Régions détectées</h3>
                    <p><strong>{{ report.text_detection.detected_regions|length }}</strong></p>
                </div>
            </div>

            <!-- Classification détaillée -->
            <div class="section info">
                <h2> Classification par Vision par Ordinateur</h2>
                <p><strong>Type de document identifié :</strong> {{ report.classification.type }}</p>
                <p><strong>Niveau de confiance :</strong> {{ "%.2f"|format(report.classification.confidence * 100) }}%</p>
                <p><em>Méthode : Analyse des proportions et détection de couleurs caractéristiques</em></p>
                
                <h3>Probabilités pour tous les types :</h3>
                <table>
                    <thead>
                        <tr><th>Type de document</th><th>Probabilité</th></tr>
                    </thead>
                    <tbody>
                        {% for doc_type, prob in report.classification.all_probabilities.items() %}
                        <tr>
                            <td>{{ doc_type }}</td>
                            <td>{{ "%.2f"|format(prob * 100) }}%</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- Détection de régions de texte -->
            <div class="section info">
                <h2> Détection de Régions avec MSER</h2>
                <p><strong>Couverture de texte :</strong> {{ "%.1f"|format(report.text_detection.text_coverage * 100) }}%</p>
                <p><strong>Nombre de régions détectées :</strong> {{ report.text_detection.detected_regions|length }}</p>
                <p><em>Algorithme : Maximally Stable Extremal Regions (OpenCV)</em></p>
                
                {% if report.text_detection.mask_path %}
                <h3>Masque de détection :</h3>
                <img src="{{ url_for('uploaded_file', filename=report.text_detection.mask_path) }}" 
                     class="image-preview" alt="Masque de détection">
                {% endif %}
                
                {% if report.text_detection.detected_regions %}
                <h3>Coordonnées des régions détectées :</h3>
                <table>
                    <thead>
                        <tr><th>Région</th><th>X</th><th>Y</th><th>Largeur</th><th>Hauteur</th></tr>
                    </thead>
                    <tbody>
                        {% for region in report.text_detection.detected_regions[:5] %}
                        <tr>
                            <td>{{ loop.index }}</td>
                            <td>{{ region.x }}</td>
                            <td>{{ region.y }}</td>
                            <td>{{ region.width }}</td>
                            <td>{{ region.height }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% endif %}
            </div>

            <!-- Reconnaissance de texte -->
            <div class="section success">
                <h2> Reconnaissance de Texte Avancée</h2>
                <p><strong>Méthode :</strong> {{ report.text_recognition.method }}</p>
                <p><strong>Confiance :</strong> {{ "%.1f"|format(report.text_recognition.confidence * 100) }}%</p>
                <p><strong>Temps de traitement :</strong> {{ report.text_recognition.processing_time }}s</p>
                <p><em>Simulation d'architecture CNN+RNN avec extraction heuristique intelligente</em></p>
            </div>

            <!-- Informations extraites -->
            <div class="section success">
                <h2> Informations Extraites</h2>
                <p><strong>Type de document :</strong> {{ report.structured_information.document_type }}</p>
                
                <table>
                    <thead>
                        <tr><th>Champ</th><th>Valeur</th></tr>
                    </thead>
                    <tbody>
                        {% for key, value in report.structured_information.items() %}
                        {% if key != 'document_type' %}
                        <tr>
                            <td>{{ key.replace('_', ' ').title() }}</td>
                            <td><strong>{{ value }}</strong></td>
                        </tr>
                        {% endif %}
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- Analyse technique détaillée -->
            <div class="section">
                <h2> Détails Techniques</h2>
                <details>
                    <summary>Afficher les données JSON complètes</summary>
                    <div class="json-view">{{ report | tojson(indent=2) }}</div>
                </details>
            </div>

            <!-- Actions -->
            <div style="text-align: center;">
                <a href="{{ url_for('index') }}" class="back-button"> Analyser un autre document</a>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    logger.info(" Démarrage du serveur OCR avancé (mode simulation)...")
    logger.info(" Interface disponible sur : http://localhost:5000")
    logger.info(" Mode : Simulation avec OpenCV (sans PyTorch)")
    app.run(debug=True, host='0.0.0.0', port=5000)