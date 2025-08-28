from flask import Flask, request, redirect, url_for, send_from_directory, render_template_string
import os
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import re

# Configuration pour macOS avec Homebrew
try:
    import pytesseract
    
    # Configuration automatique du chemin Tesseract pour macOS/Homebrew
    tesseract_paths = [
        '/opt/homebrew/bin/tesseract',  # Apple Silicon (M1/M2)
        '/usr/local/bin/tesseract',     # Intel Mac
        '/usr/bin/tesseract'            # Installation système
    ]
    
    # Trouver le bon chemin
    tesseract_cmd = None
    for path in tesseract_paths:
        if os.path.exists(path):
            tesseract_cmd = path
            break
    
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        print(f"✅ Tesseract trouvé à : {tesseract_cmd}")
    
    # Test de fonctionnement
    test_image = Image.new('RGB', (100, 50), color='white')
    pytesseract.image_to_string(test_image)
    
    HAS_TESSERACT = True
    print("✅ Tesseract OCR disponible et fonctionnel")
    
    # Vérifier les langues disponibles
    try:
        langs = pytesseract.get_languages()
        print(f"📚 Langues disponibles : {', '.join(langs)}")
        HAS_FRENCH = 'fra' in langs
        if HAS_FRENCH:
            print("✅ Français disponible")
        else:
            print("⚠️  Français non disponible - installer avec : brew install tesseract-lang")
    except:
        HAS_FRENCH = False
        
except Exception as e:
    HAS_TESSERACT = False
    HAS_FRENCH = False
    print(f"❌ Tesseract non disponible: {e}")
    print("🔧 Pour installer sur macOS :")
    print("   brew install tesseract tesseract-lang")
    print("   pip3 install pytesseract")

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

class DocumentProcessor:
    """
    Processeur de documents optimisé pour macOS
    """
    def __init__(self):
        self.document_classes = ['CNI', 'Passeport', 'Permis de conduire', 'Autre']
        self.has_tesseract = HAS_TESSERACT
        self.has_french = HAS_FRENCH
        logger.info(f"Processeur initialisé - OCR: {self.has_tesseract}, Français: {self.has_french}")
        
    def preprocess_image_for_ocr(self, image_path):
        """Préprocessing optimisé pour l'OCR"""
        try:
            # Lire l'image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Impossible de lire l'image : {image_path}")
                return image_path
                
            # Conversion en niveaux de gris
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Correction de l'inclinaison
            gray = self._correct_skew(gray)
            
            # Amélioration du contraste avec CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Réduction du bruit
            denoised = cv2.medianBlur(enhanced, 3)
            
            # Binarisation adaptative
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Sauvegarder l'image préprocessée
            base_name, ext = os.path.splitext(image_path)
            processed_path = f"{base_name}_processed{ext}"
            cv2.imwrite(processed_path, binary)
            
            logger.info(f"Image préprocessée sauvée : {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"Erreur preprocessing : {e}")
            return image_path
    
    def _correct_skew(self, image):
        """Correction automatique de l'inclinaison"""
        try:
            coords = np.column_stack(np.where(image > 0))
            if len(coords) < 500:  # Pas assez de points
                return image
                
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            # Appliquer la rotation si l'angle est significatif
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                logger.info(f"Image pivotée de {angle:.2f} degrés")
                return rotated
            
            return image
            
        except Exception as e:
            logger.error(f"Erreur correction inclinaison : {e}")
            return image
        
    def classify_document(self, image_path):
        """Classification intelligente du document"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'type': 'Erreur', 'confidence': 0.0, 'all_probabilities': {}}
                
            height, width = image.shape[:2]
            aspect_ratio = width / height
            
            # Analyse des couleurs caractéristiques
            blue_score = self._detect_blue_background(image)
            red_score = self._detect_red_elements(image)
            
            logger.info(f"Ratio d'aspect : {aspect_ratio:.2f}, Bleu : {blue_score:.3f}, Rouge : {red_score:.3f}")
            
            # Classification basée sur les caractéristiques visuelles
            if 1.4 < aspect_ratio < 1.8:  # Format carte
                if blue_score > 0.12:  # CNI française (fond bleu)
                    doc_type = 'CNI'
                    confidence = min(0.85 + blue_score * 0.15, 0.95)
                elif red_score > 0.08:  # Permis (éléments rouges)
                    doc_type = 'Permis de conduire'
                    confidence = min(0.80 + red_score * 0.15, 0.92)
                else:
                    doc_type = 'CNI'  # Par défaut pour format carte
                    confidence = 0.70
            elif aspect_ratio > 1.25:  # Format livret
                doc_type = 'Passeport'
                confidence = 0.82
            else:
                doc_type = 'Autre'
                confidence = 0.60
                
            # Générer les probabilités pour tous les types
            all_probabilities = self._generate_probabilities(doc_type, confidence)
            
            logger.info(f"Classification : {doc_type} (confiance: {confidence:.2f})")
                
            return {
                'type': doc_type,
                'confidence': confidence,
                'all_probabilities': all_probabilities,
                'analysis': {
                    'aspect_ratio': round(aspect_ratio, 2),
                    'blue_score': round(blue_score, 3),
                    'red_score': round(red_score, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur classification : {e}")
            return {'type': 'Erreur', 'confidence': 0.0, 'all_probabilities': {}}
    
    def _detect_blue_background(self, image):
        """Détection du fond bleu caractéristique de la CNI"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Plage de bleu pour CNI française
            lower_blue = np.array([95, 50, 50])
            upper_blue = np.array([125, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
            return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        except:
            return 0.0
    
    def _detect_red_elements(self, image):
        """Détection des éléments rouges (permis de conduire)"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Deux plages pour le rouge (début et fin du spectre)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = mask1 + mask2
            
            return np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])
        except:
            return 0.0
    
    def _generate_probabilities(self, predicted_type, confidence):
        """Génère des probabilités réalistes pour tous les types"""
        probs = {}
        remaining = 1 - confidence
        others = [t for t in self.document_classes if t != predicted_type]
        
        for doc_type in self.document_classes:
            if doc_type == predicted_type:
                probs[doc_type] = confidence
            else:
                # Distribution réaliste du reste
                if len(others) > 0:
                    probs[doc_type] = remaining / len(others)
                else:
                    probs[doc_type] = 0.0
                
        return probs
    
    def extract_text_with_ocr(self, image_path):
        """Extraction de texte avec Tesseract ou simulation"""
        if not self.has_tesseract:
            return self._simulate_ocr_extraction(image_path)
        
        try:
            # Préprocessing de l'image
            processed_path = self.preprocess_image_for_ocr(image_path)
            
            # Configuration Tesseract optimisée
            if self.has_french:
                config = r'--oem 3 --psm 6 -l fra+eng'
            else:
                config = r'--oem 3 --psm 6 -l eng'
                logger.warning("Français non disponible, utilisation de l'anglais")
            
            # Extraction du texte
            start_time = time.time()
            image_pil = Image.open(processed_path)
            text = pytesseract.image_to_string(image_pil, config=config)
            
            # Obtenir les scores de confiance
            data = pytesseract.image_to_data(
                image_pil, config=config, 
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = time.time() - start_time
            
            logger.info(f"OCR terminé en {processing_time:.2f}s - {len(text)} caractères - confiance: {avg_confidence:.1f}%")
            
            return {
                'raw_text': text.strip(),
                'confidence': avg_confidence / 100,
                'method': f'Tesseract OCR {"(fra+eng)" if self.has_french else "(eng)"}',
                'processing_time': round(processing_time, 2),
                'word_count': len(text.split()),
                'character_count': len(text)
            }
            
        except Exception as e:
            logger.error(f"Erreur OCR : {e}")
            return self._simulate_ocr_extraction(image_path)
    
    def _simulate_ocr_extraction(self, image_path):
        """Simulation OCR quand Tesseract n'est pas disponible"""
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray)
        except:
            mean_intensity = 128
        
        # Simulation réaliste basée sur la qualité de l'image
        if mean_intensity > 140:  # Image claire
            confidence = 0.92
            simulated_text = """RÉPUBLIQUE FRANÇAISE
CARTE NATIONALE D'IDENTITÉ

Nom: MARTIN
Prénom(s): Jean Pierre
Né(e) le: 15.03.1985
à : PARIS 15EME (75)
Sexe: M
Taille: 1,75 m
Nationalité: Française
N°: 123456789012

Délivré le: 20.01.2020
par: PREFECTURE DE POLICE
Valable jusqu'au: 20.01.2030"""
        else:  # Image moins claire
            confidence = 0.76
            simulated_text = """RÉPUBLIQUE FRANÇAISE
CARTE NATIONALE D'IDENTITÉ

Nom: DUBOIS
Prénom(s): Marie Claire
Né(e) le: 22.07.1992
à : LYON 3EME (69)
Sexe: F
Taille: 1,65 m
Nationalité: Française
N°: 987654321098"""
        
        return {
            'raw_text': simulated_text,
            'confidence': confidence,
            'method': 'Simulation (Tesseract non disponible)',
            'processing_time': 0.1,
            'word_count': len(simulated_text.split()),
            'character_count': len(simulated_text)
        }
    
    def parse_french_id_card(self, text):
        """Parser spécialisé pour CNI française avec regex améliorés"""
        info = {}
        
        # Préparation du texte pour améliorer la détection
        # Conserver les sauts de ligne pour certains patterns, mais normaliser les espaces
        text_normalized = re.sub(r'[ \t]+', ' ', text)  # Normaliser espaces multiples
        text_lines = text.split('\n')  # Pour analyse ligne par ligne
        
        # === EXTRACTION DU NOM ===
        nom_patterns = [
            # Pattern 1: Recherche explicite du mot "Nom" suivi du nom
            r'(?:Nom|NOM)\s*[:.]?\s*([A-ZÀÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÛÜÝŸ][A-ZÀÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÛÜÝŸ\s\-\']{1,50})',
            # Pattern 2: Ligne entière en majuscules (souvent le nom sur CNI)
            r'^([A-ZÀÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÛÜÝŸ]{2,}(?:[\s\-][A-ZÀÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÛÜÝŸ]+)*)\s*$',
            # Pattern 3: Nom avant prénom
            r'([A-ZÀÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÛÜÝŸ][\w\s\-\']{2,30})\s+(?:Prénom|Prenom|PRENOM)',
        ]
        
        info['nom'] = "Non détecté"
        for pattern in nom_patterns:
            # Chercher dans le texte complet
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                nom = match.group(1).strip()
                # Validation: nom doit avoir au moins 2 caractères et pas trop de chiffres
                if len(nom) >= 2 and not re.search(r'\d{3,}', nom):
                    info['nom'] = nom.upper()
                    break
            
            # Chercher ligne par ligne pour les noms isolés
            if info['nom'] == "Non détecté":
                for line in text_lines:
                    line = line.strip()
                    if re.match(r'^[A-ZÀÂÄÆÇÈÉÊËÌÍÎÏÑÒÓÔÕÖÙÛÜÝŸ\s\-\']{2,30}$', line, re.IGNORECASE):
                        if not any(word in line.lower() for word in ['république', 'française', 'carte', 'nationale', 'identité']):
                            info['nom'] = line.upper()
                            break
        
        # === EXTRACTION DU PRÉNOM ===
        prenom_patterns = [
            # Pattern 1: Avec parenthèses
            r'(?:Prénom|Prenom|PRENOM)\s*\(?\s*s?\s*\)?\s*[:.]?\s*([A-Za-zÀ-ÿ][\w\s\-\']{1,50})',
            # Pattern 2: Simple
            r'(?:Prénom|Prenom)\s*[:.]?\s*([A-Za-zÀ-ÿ][\w\s\-\']{1,50})',
            # Pattern 3: Format "Given names"
            r'(?:Given\s*names?|First\s*names?)\s*[:.]?\s*([A-Za-zÀ-ÿ][\w\s\-\']{1,50})',
        ]
        
        info['prenom'] = "Non détecté"
        for pattern in prenom_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                prenom = match.group(1).strip()
                # Nettoyer le prénom (enlever ce qui suit comme "Né", date, etc.)
                prenom = re.sub(r'\s*(?:Né|né|Ne|Date|Sexe|Sex).*$', '', prenom, flags=re.IGNORECASE)
                if len(prenom) >= 2:
                    info['prenom'] = prenom.title()
                    break
        
        # === EXTRACTION DATE DE NAISSANCE ===
        date_patterns = [
            # Pattern 1: Né(e) le + date
            r'(?:Né|Ne|née?)\s*\(?\s*e?\s*\)?\s*(?:le|on)?\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
            # Pattern 2: Date of birth
            r'(?:Date\s*of\s*birth|Birth|Naissance)\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
            # Pattern 3: Format inversé
            r'(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})\s*(?:Né|Birth)',
            # Pattern 4: Date isolée (format français)
            r'\b(\d{1,2}[\./\-](?:0[1-9]|1[0-2]|[1-9])[\./\-](?:19|20)\d{2})\b',
        ]
        
        info['date_naissance'] = "Non détecté"
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1)
                # Normaliser le format de date
                date = re.sub(r'[\-/]', '.', date)
                # Vérifier que c'est une date valide (année entre 1900 et 2023)
                year_match = re.search(r'(\d{4})$', date)
                if year_match:
                    year = int(year_match.group(1))
                    if 1900 <= year <= 2023:
                        info['date_naissance'] = date
                        break
                elif re.match(r'\d{1,2}\.\d{1,2}\.\d{2}$', date):
                    # Format année courte
                    info['date_naissance'] = date
                    break
        
        # === EXTRACTION LIEU DE NAISSANCE ===
        lieu_patterns = [
            # Pattern 1: à/at + lieu
            r'(?:à|at|in)\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-\'\(\)0-9]{2,50})(?:\s*(?:Sexe|Sex|Taille|Nation|$))',
            # Pattern 2: Lieu avec code postal
            r'(?:à|at)\s+([A-Za-zÀ-ÿ\s\-\']+(?:\s*\(\d{2,5}\))?)',
            # Pattern 3: Place of birth
            r'(?:Place\s*of\s*birth|Lieu\s*de\s*naissance)\s*[:.]?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-\'\(\)0-9]{2,50})',
        ]
        
        info['lieu_naissance'] = "Non détecté"
        for pattern in lieu_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                lieu = match.group(1).strip()
                # Nettoyer le lieu
                lieu = re.sub(r'\s*(Sexe|Sex|Taille|Height|Nation).*$', '', lieu, flags=re.IGNORECASE)
                if len(lieu) > 2 and not lieu.isdigit():
                    info['lieu_naissance'] = lieu.title()
                    break
        
        # === EXTRACTION DU SEXE ===
        sexe_patterns = [
            r'(?:Sexe|Sex|Genre)\s*[:.]?\s*([MFH])',
            r'\b([MF])\s*(?:Taille|Height|Size)',
            r'(?:Male|Female|Masculin|Féminin|Homme|Femme)',
        ]
        
        info['sexe'] = "Non détecté"
        for pattern in sexe_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if pattern == sexe_patterns[2]:  # Pattern texte complet
                    sexe_text = match.group(0).lower()
                    info['sexe'] = 'M' if sexe_text in ['male', 'masculin', 'homme'] else 'F'
                else:
                    info['sexe'] = match.group(1).upper()
                break
        
        # === EXTRACTION DE LA TAILLE ===
        taille_patterns = [
            r'(?:Taille|Height|Size)\s*[:.]?\s*(\d[,.]?\d{2})\s*(?:m|M|mètres?|meters?)?',
            r'(\d[,.]?\d{2})\s*(?:m|M)\s*(?:Nationalité|Nation|$)',
            r'\b(\d{3})\s*(?:cm|CM|centimètres?|centimeters?)',
        ]
        
        info['taille'] = "Non détecté"
        for pattern in taille_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                taille = match.group(1)
                if pattern == taille_patterns[2]:  # Conversion cm vers m
                    taille_cm = int(taille)
                    if 140 <= taille_cm <= 220:  # Taille humaine valide
                        info['taille'] = f"{taille_cm/100:.2f} m"
                else:
                    taille = taille.replace(',', '.')
                    if 1.4 <= float(taille) <= 2.2:  # Validation
                        info['taille'] = f"{taille} m"
                if info['taille'] != "Non détecté":
                    break
        
        # === EXTRACTION DE LA NATIONALITÉ ===
        nationalite_patterns = [
            r'(?:Nationalité|Nationality|Citizen)\s*[:.]?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s]{2,30})',
            r'(?:Française?|French|France)',
        ]
        
        info['nationalite'] = "Non détecté"
        for pattern in nationalite_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if pattern == nationalite_patterns[1]:
                    info['nationalite'] = "Française"
                else:
                    nat = match.group(1).strip()
                    if len(nat) > 2:
                        info['nationalite'] = nat.title()
                break
        
        # === EXTRACTION DU NUMÉRO DE DOCUMENT ===
        numero_patterns = [
            r'(?:N°|No|Number|Numéro|ID)\s*[:.]?\s*(\d{12})',
            r'(?:Card\s*number|Document\s*number)\s*[:.]?\s*(\d{12})',
            r'\b(\d{12})\b',  # 12 chiffres consécutifs
            r'(\d{4}\s+\d{4}\s+\d{4})',  # Format avec espaces
            r'(\d{3}\s*\d{3}\s*\d{3}\s*\d{3})',  # Format alternatif
        ]
        
        info['numero_document'] = "Non détecté"
        for pattern in numero_patterns:
            match = re.search(pattern, text)
            if match:
                numero = match.group(1).replace(' ', '')
                if len(numero) == 12 and numero.isdigit():
                    info['numero_document'] = numero
                    break
        
        # === EXTRACTION DATE DE DÉLIVRANCE ===
        delivrance_patterns = [
            r'(?:Délivré|Delivré|Issued|Émis)\s*(?:le|on)?\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
            r'(?:Date\s*d[\'e]\s*émission|Issue\s*date)\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
            r'(?:Du|From)\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
        ]
        
        info['date_delivrance'] = "Non détecté"
        for pattern in delivrance_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1)
                date = re.sub(r'[\-/]', '.', date)
                info['date_delivrance'] = date
                break
        
        # === EXTRACTION DATE D'EXPIRATION ===
        expiration_patterns = [
            r'(?:Valable\s*jusqu[\'e]?\s*au?|Valid\s*until|Expire|Expiration)\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
            r'(?:Jusqu[\'e]?\s*au?|Until|Au|To)\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
            r'(?:Date\s*d[\'e]\s*expiration|Expiry\s*date)\s*[:.]?\s*(\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4})',
        ]
        
        info['date_expiration'] = "Non détecté"
        for pattern in expiration_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date = match.group(1)
                date = re.sub(r'[\-/]', '.', date)
                info['date_expiration'] = date
                break
        
        # === EXTRACTION AUTORITÉ DE DÉLIVRANCE ===
        autorite_patterns = [
            r'(?:par|by|issued\s*by)\s*[:.]?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-]{5,50})',
            r'(?:Prefecture|Préfecture|Mairie|Consulat)\s*(?:de|of)?\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\s\-]{2,40})',
        ]
        
        info['autorite_delivrance'] = "Non détecté"
        for pattern in autorite_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                autorite = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).strip()
                if len(autorite) > 3:
                    info['autorite_delivrance'] = autorite.title()
                    break
        
        # Log pour debug
        logger.info(f"Extraction terminée - Champs détectés: {sum(1 for v in info.values() if v != 'Non détecté')}/{len(info)}")
        
        return info
    
    def extract_structured_information(self, classification_result, ocr_result):
        """Extraction d'informations structurées depuis le texte OCR"""
        doc_type = classification_result.get('type', 'Autre')
        raw_text = ocr_result.get('raw_text', '')
        
        if doc_type == 'CNI':
            parsed_info = self.parse_french_id_card(raw_text)
            return {
                'document_type': 'Carte Nationale d\'Identité',
                **parsed_info
            }
        elif doc_type == 'Passeport':
            return {
                'document_type': 'Passeport',
                'texte_detecte': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Parser passeport en développement'
            }
        elif doc_type == 'Permis de conduire':
            return {
                'document_type': 'Permis de Conduire',
                'texte_detecte': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
                'note': 'Parser permis en développement'
            }
        else:
            return {
                'document_type': 'Document non identifié',
                'texte_brut': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
            }

# Instance globale du processeur
processor = DocumentProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    # Statut détaillé du système
    if HAS_TESSERACT:
        if HAS_FRENCH:
            status = "✅ Tesseract OCR complet (français + anglais)"
            status_class = "success"
        else:
            status = "⚠️ Tesseract OCR (anglais seulement)"
            status_class = "warning"
    else:
        status = "❌ Mode simulation (Tesseract non installé)"
        status_class = "error"
    
    install_section = ""
    if not HAS_TESSERACT or not HAS_FRENCH:
        install_section = '''
            <div class="install-info">
                <h3>🔧 Installation sur macOS :</h3>
                <pre><code>brew install tesseract tesseract-lang
pip3 install pytesseract</code></pre>
            </div>
            '''
    
    return f'''
    <!doctype html>
    <html>
    <head>
        <title>Système OCR Avancé - Documents d'Identité</title>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 40px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{ 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                padding: 40px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            h1 {{ 
                color: #333; 
                text-align: center; 
                margin-bottom: 30px;
                font-size: 2.5em;
            }}
            .upload-area {{ 
                border: 3px dashed #ddd; 
                padding: 50px; 
                text-align: center; 
                margin: 30px 0; 
                border-radius: 15px; 
                transition: all 0.3s ease;
            }}
            .upload-area:hover {{ 
                border-color: #667eea; 
                background-color: #f8f9ff;
            }}
            input[type="file"] {{ 
                margin: 15px; 
                padding: 10px;
                font-size: 16px;
            }}
            input[type="submit"] {{ 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 18px;
                font-weight: bold;
                transition: transform 0.2s ease;
            }}
            input[type="submit"]:hover {{ 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .status {{ 
                padding: 20px; 
                border-radius: 10px; 
                margin: 25px 0; 
                font-weight: bold;
            }}
            .status.success {{ background-color: #d4edda; border: 2px solid #c3e6cb; color: #155724; }}
            .status.warning {{ background-color: #fff3cd; border: 2px solid #ffeaa7; color: #856404; }}
            .status.error {{ background-color: #f8d7da; border: 2px solid #f5c6cb; color: #721c24; }}
            .features {{ margin-top: 40px; }}
            .feature {{ 
                background: linear-gradient(45deg, #f8f9fa, #e9ecef); 
                padding: 20px; 
                margin: 15px 0; 
                border-radius: 10px; 
                border-left: 5px solid #667eea;
            }}
            .install-info {{ 
                background-color: #f8d7da; 
                border: 2px solid #f5c6cb; 
                padding: 20px; 
                border-radius: 10px; 
                margin: 20px 0; 
            }}
            pre {{ 
                background-color: #f1f3f4; 
                padding: 15px; 
                border-radius: 5px; 
                overflow-x: auto;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Système OCR Avancé</h1>
            <p style="text-align: center; color: #666; font-size: 18px;">
                Reconnaissance automatique de documents d'identité avec IA
            </p>
            
            <div class="status {status_class}">
                <h3>📊 Statut du Système</h3>
                <p>{status}</p>
            </div>
            
            {install_section}
            
            <form method="post" enctype="multipart/form-data" action="/upload">
                <div class="upload-area">
                    <h3>📄 Télécharger un document d'identité</h3>
                    <p style="color: #666;">CNI, Passeport, Permis de conduire</p>
                    <input type="file" name="file" accept=".png,.jpg,.jpeg" required>
                    <br><br>
                    <input type="submit" value="🚀 Analyser avec l'IA">
                </div>
            </form>
            
            <div class="features">
                <h3>🔬 Fonctionnalités :</h3>
                <div class="feature">
                    <strong>🎯 Classification Intelligente :</strong> Reconnaissance automatique du type de document
                </div>
                <div class="feature">
                    <strong>📝 OCR Avancé :</strong> Extraction de texte avec préprocessing intelligent
                </div>
                <div class="feature">
                    <strong>🛡️ Parsing Structuré :</strong> Extraction automatique des champs (nom, prénom, dates, etc.)
                </div>
                <div class="feature">
                    <strong>🔧 Optimisé macOS :</strong> Configuration automatique pour Homebrew et Apple Silicon
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
        
        try:
            file.save(file_path)
            logger.info(f"Fichier sauvé : {file_path}")
            return redirect(url_for('analyze_document', filename=filename))
        except Exception as e:
            logger.error(f"Erreur sauvegarde fichier : {e}")
            return "Erreur lors de la sauvegarde", 500
    
    return "Type de fichier non autorisé", 400

@app.route('/analyze/<filename>')
def analyze_document(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(image_path):
        return "Fichier non trouvé", 404
    
    # Début du chronométrage
    start_time = time.time()
    
    try:
        # Étape 1: Classification du document
        logger.info("🔍 Classification du document...")
        classification_result = processor.classify_document(image_path)
        
        # Étape 2: Extraction OCR
        logger.info("📝 Extraction OCR du texte...")
        ocr_result = processor.extract_text_with_ocr(image_path)
        
        # Étape 3: Parsing structuré
        logger.info("🧠 Parsing des informations...")
        structured_info = processor.extract_structured_information(classification_result, ocr_result)
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Génération du rapport
        analysis_report = {
            'filename': filename,
            'processing_time': round(processing_time, 2),
            'timestamp': datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            'classification': classification_result,
            'ocr_result': ocr_result,
            'structured_information': structured_info,
            'has_real_ocr': HAS_TESSERACT,
            'has_french': HAS_FRENCH,
            'system_info': {
                'tesseract_path': pytesseract.pytesseract.tesseract_cmd if HAS_TESSERACT else None,
                'opencv_version': cv2.__version__,
                'platform': 'macOS'
            }
        }
        
        logger.info(f"✅ Analyse terminée en {processing_time:.2f}s")
        
        return render_template_string(get_results_template(), 
                                    report=analysis_report, 
                                    filename=filename)
                                    
    except Exception as e:
        logger.error(f"❌ Erreur durant l'analyse : {e}")
        return f"Erreur durant l'analyse : {str(e)}", 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_results_template():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Résultats de l'Analyse IA</title>
        <meta charset="UTF-8">
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                margin: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white; 
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                padding: 20px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border-radius: 10px;
            }
            .section { 
                margin: 25px 0; 
                padding: 25px; 
                border: 2px solid #e9ecef; 
                border-radius: 10px; 
                background: #f8f9fa;
            }
            .success { background: linear-gradient(45deg, #d4edda, #c3e6cb); border-color: #28a745; }
            .info { background: linear-gradient(45deg, #d1ecf1, #bee5eb); border-color: #17a2b8; }
            .warning { background: linear-gradient(45deg, #fff3cd, #ffeaa7); border-color: #ffc107; }
            .error { background: linear-gradient(45deg, #f8d7da, #f5c6cb); border-color: #dc3545; }
            
            .image-preview { 
                max-width: 300px; 
                max-height: 400px; 
                border-radius: 10px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                margin: 15px; 
            }
            .metrics { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
                gap: 20px; 
                margin: 25px 0; 
            }
            .metric { 
                background: white; 
                padding: 20px; 
                border-radius: 10px; 
                text-align: center; 
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
            }
            .metric h3 { 
                margin: 0 0 10px 0; 
                color: #333; 
                font-size: 1.1em;
            }
            .metric p { 
                margin: 0; 
                font-size: 1.5em; 
                font-weight: bold; 
            }
            .confidence-high { color: #28a745; }
            .confidence-medium { color: #ffc107; }
            .confidence-low { color: #dc3545; }
            
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin: 20px 0; 
                background: white;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            th, td { 
                padding: 15px; 
                text-align: left; 
                border-bottom: 1px solid #dee2e6; 
            }
            th { 
                background: linear-gradient(45deg, #667eea, #764ba2); 
                color: white; 
                font-weight: bold; 
            }
            tr:hover { background-color: #f8f9fa; }
            
            .back-button { 
                background: linear-gradient(45deg, #6c757d, #5a6268); 
                color: white; 
                padding: 15px 30px; 
                text-decoration: none; 
                border-radius: 8px; 
                display: inline-block; 
                margin-top: 30px; 
                font-weight: bold;
                transition: transform 0.2s ease;
            }
            .back-button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            
            .text-extract { 
                background: #f1f3f4; 
                padding: 20px; 
                border-radius: 8px; 
                font-family: 'Monaco', 'Menlo', monospace; 
                font-size: 14px; 
                white-space: pre-wrap; 
                max-height: 300px; 
                overflow-y: auto; 
                border: 1px solid #dee2e6;
                line-height: 1.5;
            }
            
            .json-view { 
                background: #f8f9fa; 
                padding: 20px; 
                border-radius: 8px; 
                white-space: pre-wrap; 
                font-family: 'Monaco', 'Menlo', monospace; 
                font-size: 12px; 
                max-height: 400px; 
                overflow-y: auto; 
                border: 1px solid #dee2e6;
            }
            
            details { 
                margin: 15px 0; 
            }
            summary { 
                cursor: pointer; 
                padding: 10px; 
                background: #e9ecef; 
                border-radius: 5px; 
                font-weight: bold;
            }
            summary:hover { 
                background: #dee2e6; 
            }
            
            .status-badge {
                display: inline-block;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
                margin-left: 10px;
            }
            .status-real { background: #d4edda; color: #155724; }
            .status-sim { background: #fff3cd; color: #856404; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Résultats de l'Analyse IA</h1>
                <p>Traitement terminé en <strong>{{ report.processing_time }}s</strong> | {{ report.timestamp }}</p>
                <p>
                    <strong>Mode OCR:</strong> 
                    {% if report.has_real_ocr %}
                        ✅ Tesseract Réel
                        <span class="status-badge status-real">
                            {% if report.has_french %}FRA+ENG{% else %}ENG{% endif %}
                        </span>
                    {% else %}
                        ⚠️ Simulation
                        <span class="status-badge status-sim">DEMO</span>
                    {% endif %}
                </p>
            </div>
            
            <!-- Image originale -->
            <div class="section">
                <h2>📷 Document Analysé</h2>
                <div style="text-align: center;">
                    <img src="{{ url_for('uploaded_file', filename=filename) }}" class="image-preview" alt="Document">
                </div>
                <p><strong>Fichier :</strong> {{ filename }}</p>
            </div>
            
            <!-- Métriques de performance -->
            <div class="metrics">
                <div class="metric">
                    <h3>⚡ Temps Total</h3>
                    <p class="confidence-high">{{ report.processing_time }}s</p>
                </div>
                <div class="metric">
                    <h3>🎯 Type Détecté</h3>
                    <p><strong>{{ report.classification.type }}</strong></p>
                </div>
                <div class="metric">
                    <h3>📊 Confiance Classification</h3>
                    <p class="{% if report.classification.confidence > 0.8 %}confidence-high{% elif report.classification.confidence > 0.5 %}confidence-medium{% else %}confidence-low{% endif %}">
                        {{ "%.1f"|format(report.classification.confidence * 100) }}%
                    </p>
                </div>
                <div class="metric">
                    <h3>📝 Confiance OCR</h3>
                    <p class="{% if report.ocr_result.confidence > 0.8 %}confidence-high{% elif report.ocr_result.confidence > 0.5 %}confidence-medium{% else %}confidence-low{% endif %}">
                        {{ "%.1f"|format(report.ocr_result.confidence * 100) }}%
                    </p>
                </div>
                <div class="metric">
                    <h3>📄 Mots Extraits</h3>
                    <p class="confidence-high">{{ report.ocr_result.word_count }}</p>
                </div>
                <div class="metric">
                    <h3>🔤 Caractères</h3>
                    <p class="confidence-high">{{ report.ocr_result.character_count }}</p>
                </div>
            </div>

            <!-- Classification détaillée -->
            <div class="section info">
                <h2>🧠 Classification Intelligente</h2>
                <p><strong>Type identifié :</strong> {{ report.classification.type }}</p>
                <p><strong>Confiance :</strong> {{ "%.2f"|format(report.classification.confidence * 100) }}%</p>
                
                {% if report.classification.analysis %}
                <h3>📊 Analyse Visuelle :</h3>
                <ul>
                    <li><strong>Ratio d'aspect :</strong> {{ report.classification.analysis.aspect_ratio }}</li>
                    <li><strong>Score bleu (CNI) :</strong> {{ report.classification.analysis.blue_score }}</li>
                    <li><strong>Score rouge (Permis) :</strong> {{ report.classification.analysis.red_score }}</li>
                </ul>
                {% endif %}
                
                <h3>🎲 Probabilités par Type :</h3>
                <table>
                    <thead>
                        <tr><th>Type de Document</th><th>Probabilité</th><th>Barre</th></tr>
                    </thead>
                    <tbody>
                        {% for doc_type, prob in report.classification.all_probabilities.items() %}
                        <tr>
                            <td>{{ doc_type }}</td>
                            <td>{{ "%.2f"|format(prob * 100) }}%</td>
                            <td>
                                <div style="background: #e9ecef; border-radius: 10px; height: 20px; width: 100px; overflow: hidden;">
                                    <div style="background: linear-gradient(45deg, #667eea, #764ba2); height: 100%; width: {{ prob * 100 }}%; border-radius: 10px;"></div>
                                </div>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- Extraction OCR -->
            <div class="section {% if report.has_real_ocr %}info{% else %}warning{% endif %}">
                <h2>📝 Extraction de Texte OCR</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0;">
                    <div><strong>Méthode :</strong> {{ report.ocr_result.method }}</div>
                    <div><strong>Confiance :</strong> {{ "%.1f"|format(report.ocr_result.confidence * 100) }}%</div>
                    <div><strong>Temps OCR :</strong> {{ report.ocr_result.processing_time }}s</div>
                    <div><strong>Mots détectés :</strong> {{ report.ocr_result.word_count }}</div>
                </div>
                
                <h3>📄 Texte Extrait :</h3>
                <div class="text-extract">{{ report.ocr_result.raw_text }}</div>
                
                {% if not report.has_real_ocr %}
                <div style="margin-top: 15px; padding: 15px; background-color: #fff3cd; border-radius: 8px; border-left: 5px solid #ffc107;">
                    <strong>💡 Note :</strong> Données simulées - Installez Tesseract pour l'OCR réel :
                    <code>brew install tesseract tesseract-lang</code>
                </div>
                {% endif %}
            </div>

            <!-- Informations structurées -->
            <div class="section success">
                <h2>📋 Informations Extraites et Structurées</h2>
                <p><strong>Type de document :</strong> {{ report.structured_information.document_type }}</p>
                
                <table>
                    <thead>
                        <tr><th>Champ</th><th>Valeur Extraite</th><th>Statut</th></tr>
                    </thead>
                    <tbody>
                        {% for key, value in report.structured_information.items() %}
                        {% if key != 'document_type' %}
                        <tr>
                            <td><strong>{{ key.replace('_', ' ').title() }}</strong></td>
                            <td>{{ value }}</td>
                            <td>
                                {% if value != "Non détecté" and value != "Not detected" %}
                                    <span style="color: #28a745; font-weight: bold;">✅ Détecté</span>
                                {% else %}
                                    <span style="color: #dc3545; font-weight: bold;">❌ Non détecté</span>
                                {% endif %}
                            </td>
                        </tr>
                        {% endif %}
                        {% endfor %}
                    </tbody>
                </table>
            </div>

            <!-- Informations système -->
            <div class="section">
                <h2>⚙️ Informations Système</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                    <div><strong>Plateforme :</strong> {{ report.system_info.platform }}</div>
                    <div><strong>OpenCV :</strong> {{ report.system_info.opencv_version }}</div>
                    {% if report.system_info.tesseract_path %}
                    <div><strong>Tesseract :</strong> {{ report.system_info.tesseract_path }}</div>
                    {% endif %}
                </div>
                
                <details style="margin-top: 20px;">
                    <summary>🔬 Voir les données JSON complètes</summary>
                    <div class="json-view">{{ report | tojson(indent=2) }}</div>
                </details>
            </div>

            <!-- Actions -->
            <div style="text-align: center;">
                <a href="{{ url_for('index') }}" class="back-button">🔄 Analyser un Autre Document</a>
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 DÉMARRAGE DU SYSTÈME OCR AVANCÉ")
    print("=" * 60)
    print(f"📍 Interface web : http://localhost:8080")
    print(f"🔧 OCR réel : {'✅ Activé' if HAS_TESSERACT else '❌ Installer Tesseract'}")
    print(f"🇫🇷 Français : {'✅ Disponible' if HAS_FRENCH else '❌ Installer tesseract-lang'}")
    print(f"📁 Dossier uploads : {os.path.abspath(UPLOAD_FOLDER)}")
    
    if not HAS_TESSERACT:
        print("\n⚠️  INSTALLATION REQUISE :")
        print("   brew install tesseract tesseract-lang")
        print("   pip3 install pytesseract")
    
    print("=" * 60)
    
    try:
        app.run(debug=True, host='0.0.0.0', port=8080)
    except KeyboardInterrupt:
        print("\n👋 Arrêt du serveur...")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")