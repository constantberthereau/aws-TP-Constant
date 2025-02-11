import os
import boto3
import cv2
import nltk
from nltk.corpus import stopwords
import json
import streamlit as st
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def check_filetype(filename):
    """
    Détermine le type de fichier en fonction de son extension.
    """
    ext = os.path.splitext(filename)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        return 'image'
    elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
        return 'video'
    return None

def extract_frame_video(video_path, frame_id):
    """
    Extrait une image spécifique d'une vidéo.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    success, frame = cap.read()
    if success:
        frame_path = f"frame_{frame_id}.jpg"
        cv2.imwrite(frame_path, frame)
        cap.release()
        return frame_path
    cap.release()
    return None

def get_aws_session():
    """
    Crée et retourne une session AWS.
    """
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY")
    )

def moderate_image(image_path, client):
    """
    Détecte du contenu nécessitant une modération dans une image en utilisant un service AWS.
    """
    with open(image_path, 'rb') as image_file:
        response = client.detect_moderation_labels(Image={'Bytes': image_file.read()})
    return response.get('ModerationLabels', [])

def get_text_from_speech(filename, transcribe, job_name, bucket_name):
    """
    Convertit de la parole en texte en utilisant AWS Transcribe.
    """
    response = transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': f's3://{bucket_name}/{filename}'},
        MediaFormat='mp3',
        LanguageCode='fr-FR'
    )
    return response

def clean_text(raw_text):
    """
    Nettoie un texte en retirant les mots vides et en normalisant les mots en minuscules.
    """
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))
    words = raw_text.lower().split()
    return ' '.join([word for word in words if word not in stop_words])

def extract_keyphrases(text, comprehend):
    """
    Extrait les expressions clés d'un texte.
    """
    response = comprehend.detect_key_phrases(Text=text, LanguageCode='fr')
    keyphrases = [phrase['Text'] for phrase in response['KeyPhrases']]
    return keyphrases[:10]

def detect_objects(image_path, client):
    """
    Détecte les objets dans une image en utilisant Amazon Rekognition.
    """
    with open(image_path, 'rb') as image_file:
        response = client.detect_labels(Image={'Bytes': image_file.read()}, MaxLabels=10)
    return [label['Name'] for label in response.get('Labels', [])]

def detect_celebrities(image_path, client):
    """
    Identifie les célébrités dans une image en utilisant le service Amazon Rekognition.
    """
    with open(image_path, 'rb') as image_file:
        response = client.recognize_celebrities(Image={'Bytes': image_file.read()})
    return [celeb['Name'] for celeb in response.get('CelebrityFaces', [])]

def detect_emotions(image_path, client):
    """
    Détecte les émotions sur les visages présents dans une image.
    """
    with open(image_path, 'rb') as image_file:
        response = client.detect_faces(Image={'Bytes': image_file.read()}, Attributes=['ALL'])
    return response.get('FaceDetails', [])

def summarize_emotions(faces_info):
    """
    Résume les émotions détectées sur tous les visages d'une image.
    """
    emotions = {}
    for face in faces_info:
        for emotion in face['Emotions']:
            emotions[emotion['Type']] = emotions.get(emotion['Type'], 0) + emotion['Confidence']
    return max(emotions, key=emotions.get) if emotions else None

def process_media(media_file, rekognition, transcribe, comprehend, s3, bucket_name):
    """
    Traite un fichier multimédia (image ou vidéo) pour modérer le contenu, détecter des objets/célébrités,
    transcrire le discours et extraire des expressions clés.
    """
    file_type = check_filetype(media_file)
    print("file_type")
    print(file_type)
    if file_type == 'image':
        moderation_labels = moderate_image(media_file, rekognition)
        print("moderation_labels")
        print(moderation_labels)
        objects = detect_objects(media_file, rekognition)
        print("objects")
        print(objects)
        celebrities = detect_celebrities(media_file, rekognition)
        print("celebrities")
        print(celebrities)
        emotions = detect_emotions(media_file, rekognition)
        print("emotions")
        print(emotions)
        return {
            'moderation': moderation_labels,
            'objects': objects,
            'celebrities': celebrities,
            'emotions': summarize_emotions(emotions)
        }
    elif file_type == 'video':
        s3.upload_file(media_file, bucket_name, os.path.basename(media_file))
        job_name = os.path.splitext(os.path.basename(media_file))[0]
        transcription = get_text_from_speech(media_file, transcribe, job_name, bucket_name)
        extracted_frame = extract_frame_video(media_file, 100)
        if extracted_frame:
            image_analysis = process_media(extracted_frame, rekognition, transcribe, comprehend, s3, bucket_name)
        cleaned_text = clean_text(transcription)
        keyphrases = extract_keyphrases(cleaned_text, comprehend)
        return {
            'transcription': cleaned_text,
            'keyphrases': keyphrases,
            'image_analysis': image_analysis
        }
    return None
