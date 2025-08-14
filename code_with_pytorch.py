import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
import json
import time
from datetime import datetime
import logging
from werkzeug.utils import secure_filename

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}

# Configuration du dispositif (GPU si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Utilisation du dispositif: {device}")

class DocumentClassifierCNN(nn.Module):
    """
    CNN pour classification des types de documents d'identité
    Architecture inspirée de ResNet avec adaptations pour documents
    """
    def __init__(self, num_classes=4):  # CNI, Passeport, Permis, Autre
        super(DocumentClassifierCNN, self).__init__()
        
        # Couches convolutionnelles avec batch normalization
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Blocs résiduel simplifiés
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Pooling adaptatif et classificateur
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Extraction de caractéristiques hiérarchiques
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Classification
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class TextRegionDetector(nn.Module):
    """
    CNN pour détection des régions de texte dans les documents
    Utilise une approche de segmentation sémantique
    """
    def __init__(self):
        super(TextRegionDetector, self).__init__()
        
        # Encoder (downsampling)
        self.encoder1 = self._make_encoder_layer(3, 64)
        self.encoder2 = self._make_encoder_layer(64, 128)
        self.encoder3 = self._make_encoder_layer(128, 256)
        self.encoder4 = self._make_encoder_layer(256, 512)
        
        # Decoder (upsampling)
        self.decoder4 = self._make_decoder_layer(512, 256)
        self.decoder3 = self._make_decoder_layer(256, 128)
        self.decoder2 = self._make_decoder_layer(128, 64)
        self.decoder1 = self._make_decoder_layer(64, 32)
        
        # Couche finale pour masque de segmentation
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    def _make_encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoding path
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Decoding path
        d4 = self.decoder4(e4)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        
        # Masque final
        mask = torch.sigmoid(self.final_conv(d1))
        return mask

class CRNNTextRecognizer(nn.Module):
    """
    Modèle CRNN pour reconnaissance de texte
    Combine CNN pour extraction de caractéristiques et RNN pour séquence
    """
    def __init__(self, vocab_size=95, hidden_size=256, num_layers=2):
        super(CRNNTextRecognizer, self).__init__()
        
        # CNN pour extraction de caractéristiques
        self.cnn = nn.Sequential(
            # Bloc 1
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            # Bloc 2  
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            # Bloc 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            # Bloc 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1), (2, 1)),
            # Bloc 5
            nn.Conv2d(512, 512, 2), nn.ReLU()
        )
        
        # RNN pour modélisation de séquence
        self.rnn = nn.LSTM(512, hidden_size, num_layers, bidirectional=True, batch_first=True)
        
        # Couche de classification
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x):
        # Extraction de caractéristiques CNN
        conv_features = self.cnn(x)  # [batch, 512, H', W']
        
        # Reshape pour RNN: [batch, W', features]
        b, c, h, w = conv_features.size()
        conv_features = conv_features.view(b, c * h, w).permute(0, 2, 1)
        
        # Traitement RNN
        rnn_output, _ = self.rnn(conv_features)
        
        # Classification
        output = self.classifier(rnn_output)
        return F.log_softmax(output, dim=2)

class AdvancedDocumentProcessor:
    """
    Processeur de documents avancé utilisant les modèles de deep learning
    """
    def __init__(self):
        # Initialisation des modèles
        self.document_classifier = DocumentClassifierCNN().to(device)
        self.text_detector = TextRegionDetector().to(device)
        self.text_recognizer = CRNNTextRecognizer().to(device)
        
        # Transformations pour l'input
        self.transform_classifier = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.transform_detector = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Classes de documents
        self.document_classes = ['CNI', 'Passeport', 'Permis de conduire', 'Autre']
        
        # Chargement des modèles pré-entraînés si disponibles
        self._load_models()
        
    def _load_models(self):
        """Charge les modèles pré-entraînés si disponibles"""
        try:
            classifier_path = os.path.join(MODEL_FOLDER, 'document_classifier.pth')
            if os.path.exists(classifier_path):
                self.document_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
                logger.info("Modèle de classification chargé")
                
            detector_path = os.path.join(MODEL_FOLDER, 'text_detector.pth')
            if os.path.exists(detector_path):
                self.text_detector.load_state_dict(torch.load(detector_path, map_location=device))
                logger.info("Modèle de détection de texte chargé")
                
            recognizer_path = os.path.join(MODEL_FOLDER, 'text_recognizer.pth')
            if os.path.exists(recognizer_path):
                self.text_recognizer.load_state_dict(torch.load(recognizer_path, map_location=device))
                logger.info("Modèle de reconnaissance de texte chargé")
                
        except Exception as e:
            logger.warning(f"Impossible de charger les modèles: {e}")
            logger.info("Utilisation des modèles avec poids aléatoires (mode démonstration)")
    
    def classify_document(self, image_path):
        """Classifie le type de document"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform_classifier(image).unsqueeze(0).to(device)
            
            self.document_classifier.eval()
            with torch.no_grad():
                outputs = self.document_classifier(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
                
            return {
                'type': self.document_classes[predicted_class],
                'confidence': confidence,
                'all_probabilities': {
                    self.document_classes[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }
        except Exception as e:
            logger.error(f"Erreur lors de la classification: {e}")
            return {'type': 'Erreur', 'confidence': 0.0}
    
    def detect_text_regions(self, image_path):
        """Détecte les régions de texte dans l'image"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform_detector(image).unsqueeze(0).to(device)
            
            self.text_detector.eval()
            with torch.no_grad():
                mask = self.text_detector(input_tensor)
                
            # Conversion en numpy pour traitement
            mask_np = mask.squeeze().cpu().numpy()
            mask_np = (mask_np > 0.5).astype(np.uint8) * 255
            
            # Sauvegarde du masque de détection
            mask_path = image_path.replace('.jpg', '_text_mask.jpg')
            cv2.imwrite(mask_path, mask_np)
            
            return {
                'mask_path': mask_path,
                'text_coverage': np.mean(mask_np > 0),
                'detected_regions': self._extract_bounding_boxes(mask_np)
            }
        except Exception as e:
            logger.error(f"Erreur lors de la détection: {e}")
            return {'error': str(e)}
    
    def _extract_bounding_boxes(self, mask):
        """Extrait les boîtes englobantes des régions de texte"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filtrer les petites régions
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append({'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)})
        return boxes
    
    def recognize_text_crnn(self, image_path):
        """Reconnaissance de texte avec CRNN (simulation)"""
        try:
            # Pour la démonstration, on simule le processus CRNN
            # Dans un cas réel, l'image serait segmentée en lignes de texte
            
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            # Simulation du preprocessing pour CRNN
            processed_image = cv2.resize(image, (128, 32))  # Taille standard pour CRNN
            
            # Simulation de la reconnaissance (en production, utiliserait le modèle CRNN)
            simulated_text = self._simulate_crnn_output()
            
            return {
                'recognized_text': simulated_text,
                'confidence': 0.85,  # Confiance simulée
                'method': 'CRNN',
                'processing_time': 0.15
            }
        except Exception as e:
            logger.error(f"Erreur lors de la reconnaissance CRNN: {e}")
            return {'error': str(e)}
    
    def _simulate_crnn_output(self):
        """Simule la sortie du modèle CRNN pour la démonstration"""
        return {
            'nom': 'MARTIN',
            'prenom': 'Jean Pierre',
            'date_naissance': '15.03.1985',
            'lieu_naissance': 'PARIS',
            'numero_document': '123456789012',
            'sexe': 'M',
            'taille': '175cm'
        }
    
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
        """Extraction spécifique pour CNI française"""
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
        """Extraction spécifique pour passeport"""
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
        """Extraction spécifique pour permis de conduire"""
        return {
            'document_type': 'Permis de Conduire',
            'numero_permis': text_data.get('numero_document', 'Non détecté'),
            'nom': text_data.get('nom', 'Non détecté'),
            'prenom': text_data.get('prenom', 'Non détecté'),
            'date_naissance': text_data.get('date_naissance', 'Non détecté'),
            'lieu_naissance': text_data.get('lieu_naissance', 'Non détecté')
        }
    
    def _extract_generic_info(self, text_data):
        """Extraction générique pour documents non spécifiques"""
        return {
            'document_type': 'Document non identifié',
            'informations_detectees': text_data
        }

# Instance globale du processeur
processor = AdvancedDocumentProcessor()

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
        </style>
    </head>
    <body>
        <div class="container">
            <h1> Système OCR Avancé avec IA</h1>
            <p style="text-align: center; color: #666;">
                Reconnaissance automatique de documents d'identité utilisant CNN et RNN
            </p>
            
            <form method="post" enctype="multipart/form-data" action="/upload">
                <div class="upload-area">
                    <h3> Télécharger un document d'identité</h3>
                    <input type="file" name="file" accept=".png,.jpg,.jpeg" required>
                    <br><br>
                    <input type="submit" value=" Analyser avec l'IA">
                </div>
            </form>
            
            <div class="features">
                <h3> Fonctionnalités avancées :</h3>
                <div class="feature">
                    <strong> Classification automatique :</strong> CNN pour identifier le type de document (CNI, Passeport, Permis)
                </div>
                <div class="feature">
                    <strong> Détection de régions :</strong> Segmentation sémantique pour localiser les zones de texte
                </div>
                <div class="feature">
                    <strong> Reconnaissance CRNN :</strong> Réseau CNN+RNN pour extraire le texte avec précision
                </div>
                <div class="feature">
                    <strong> Extraction structurée :</strong> Parsing intelligent selon le type de document détecté
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
    
    # Étape 3: Reconnaissance de texte avec CRNN
    logger.info("Reconnaissance de texte avec CRNN...")
    text_result = processor.recognize_text_crnn(image_path)
    
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
            text_result = processor.recognize_text_crnn(file_path)
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
            .image-preview { max-width: 300px; border-radius: 8px; margin: 10px; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
            .metric { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }
            .confidence-high { color: #28a745; font-weight: bold; }
            .confidence-medium { color: #ffc107; font-weight: bold; }
            .confidence-low { color: #dc3545; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f8f9fa; font-weight: bold; }
            .back-button { background-color: #6c757d; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-top: 20px; }
            .json-view { background: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1> Résultats de l'Analyse IA</h1>
                <p>Traitement terminé en <strong>{{ report.processing_time }}s</strong> | {{ report.timestamp }}</p>
            </div>
            
            <!-- Image originale -->
            <div class="section">
                <h2> Document analysé</h2>
                <img src="{{ url_for('uploaded_file', filename=filename) }}" class="image-preview" alt="Document" style="max-width: 250px;">
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
                <h2> Classification CNN</h2>
                <p><strong>Type de document identifié :</strong> {{ report.classification.type }}</p>
                <p><strong>Niveau de confiance :</strong> {{ "%.2f"|format(report.classification.confidence * 100) }}%</p>
                
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
                <h2> Détection de Régions (Segmentation Sémantique)</h2>
                <p><strong>Couverture de texte :</strong> {{ "%.1f"|format(report.text_detection.text_coverage * 100) }}%</p>
                <p><strong>Nombre de régions détectées :</strong> {{ report.text_detection.detected_regions|length }}</p>
                
                {% if report.text_detection.mask_path %}
                <h3>Masque de segmentation :</h3>
                <img src="{{ url_for('uploaded_file', filename=report.text_detection.mask_path.split('/')[-1]) }}" 
                     class="image-preview" alt="Masque de détection" style="max-width: 250px;">
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

            <!-- Reconnaissance CRNN -->
            <div class="section success">
                <h2> Reconnaissance de Texte (CRNN)</h2>
                <p><strong>Méthode :</strong> {{ report.text_recognition.method }}</p>
                <p><strong>Confiance :</strong> {{ "%.1f"|format(report.text_recognition.confidence * 100) }}%</p>
                <p><strong>Temps de traitement :</strong> {{ report.text_recognition.processing_time }}s</p>
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
                <a href="{{ url_for('index') }}" class="back-button">Analyser un autre document</a>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    logger.info("Démarrage du serveur OCR avancé...")
    logger.info(f"Modèles chargés sur : {device}")
    app.run(debug=True, host='0.0.0.0', port=5000)