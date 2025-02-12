import os
import boto3
import cv2
import nltk
from nltk.corpus import stopwords
import json
import streamlit as st
from dotenv import load_dotenv
import time
import requests

# Charger les variables d'environnement
load_dotenv()

def check_filetype(filename):
    ext = os.path.splitext(filename)[-1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif']:
        return 'image'
    elif ext in ['.mp4', '.avi', '.mkv', '.mov']:
        return 'video'
    return None

def extract_frame_video(video_path, frame_id):
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
    return boto3.Session(
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY")
    )

def moderate_image(image_path, client):
    with open(image_path, 'rb') as image_file:
        response = client.detect_moderation_labels(Image={'Bytes': image_file.read()})
    return response.get('ModerationLabels', [])

def detect_objects(image_path, client):
    with open(image_path, 'rb') as image_file:
        response = client.detect_labels(Image={'Bytes': image_file.read()}, MaxLabels=10)
    return [label['Name'] for label in response.get('Labels', [])]

def detect_celebrities(image_path, client):
    with open(image_path, 'rb') as image_file:
        response = client.recognize_celebrities(Image={'Bytes': image_file.read()})
    return [celeb['Name'] for celeb in response.get('CelebrityFaces', [])]

def detect_emotions(image_path, client):
    with open(image_path, 'rb') as image_file:
        response = client.detect_faces(Image={'Bytes': image_file.read()}, Attributes=['ALL'])
    return response.get('FaceDetails', [])

def summarize_emotions(faces_info):
    emotions = {}
    for face in faces_info:
        for emotion in face['Emotions']:
            emotions[emotion['Type']] = emotions.get(emotion['Type'], 0) + emotion['Confidence']
    return max(emotions, key=emotions.get) if emotions else None

def get_text_from_speech(filename, transcribe, job_name, bucket_name):
    """
    Lance la transcription AWS Transcribe et attend la fin du job pour récupérer la transcription.
    """
    job_uri = f's3://{bucket_name}/{filename}'
    
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='mp4',
        LanguageCode='fr-FR'
    )
    
    # Attendre que le job soit complété
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        print("⏳ En attente de la transcription...")
        time.sleep(10)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        response = requests.get(transcript_url)
        transcript_data = response.json()

        if 'results' in transcript_data and 'transcripts' in transcript_data['results']:
            return transcript_data['results']['transcripts'][0]['transcript']
    
    return None

def clean_text(raw_text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))
    words = raw_text.lower().split()
    return ' '.join([word for word in words if word not in stop_words])

def extract_keyphrases(text, comprehend):
    response = comprehend.detect_key_phrases(Text=text, LanguageCode='fr')
    keyphrases = [phrase['Text'] for phrase in response['KeyPhrases']]
    return keyphrases[:10]

def process_media(media_file, rekognition, transcribe, comprehend, s3, bucket_name):
    file_type = check_filetype(media_file)
    
    if file_type == 'image':
        moderation_labels = moderate_image(media_file, rekognition)
        objects = detect_objects(media_file, rekognition)
        celebrities = detect_celebrities(media_file, rekognition)
        emotions = detect_emotions(media_file, rekognition)
        
        return {
            'moderation': moderation_labels,
            'objects': objects,
            'celebrities': celebrities,
            'emotions': summarize_emotions(emotions)
        }

    elif file_type == 'video':
        s3.upload_file(media_file, bucket_name, os.path.basename(media_file))

        # Vérification de l'upload
        uploaded_objects = s3.list_objects_v2(Bucket=bucket_name)
        uploaded_files = [obj['Key'] for obj in uploaded_objects.get('Contents', [])]

        if os.path.basename(media_file) not in uploaded_files:
            raise ValueError(f"⚠️ Problème lors de l'upload, fichier {media_file} introuvable dans S3 !")
        else:
            print(f"✅ Fichier {media_file} bien uploadé sur S3.")

        job_name = os.path.splitext(os.path.basename(media_file))[0]
        transcription_text = get_text_from_speech(os.path.basename(media_file), transcribe, job_name, bucket_name)

        if transcription_text:
            cleaned_text = clean_text(transcription_text)
            keyphrases = extract_keyphrases(cleaned_text, comprehend)
        else:
            cleaned_text = "⚠️ Échec de la transcription"
            keyphrases = []

        extracted_frame = extract_frame_video(media_file, 100)
        image_analysis = process_media(extracted_frame, rekognition, transcribe, comprehend, s3, bucket_name) if extracted_frame else None

        return {
            'transcription': cleaned_text,
            'keyphrases': keyphrases,
            'image_analysis': image_analysis
        }

    return None
