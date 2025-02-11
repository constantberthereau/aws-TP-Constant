import streamlit as st
import boto3
import os
import tempfile
from dotenv import load_dotenv
from moderation import process_media  # Importation du module de modération

# Charger les variables d'environnement
load_dotenv()

# Configuration de la page
st.set_page_config(page_title="Content Moderator Pro", layout="wide")

# Sidebar - Configuration AWS
st.sidebar.header("⚙️ Configuration")
st.sidebar.subheader("🔑 Credentials AWS")

# Fonction pour charger les credentials depuis .env
def load_credentials():
    return {
        "access_key": os.getenv("ACCESS_KEY", ""),
        "secret_key": os.getenv("SECRET_KEY", ""),
        "bucket_name": os.getenv("BUCKET_NAME", "test-bucket-sdvnantes-2")
    }

# Initialiser session_state pour les credentials s'ils ne sont pas encore définis
if "access_key" not in st.session_state:
    credentials = load_credentials()
    st.session_state.access_key = credentials["access_key"]
    st.session_state.secret_key = credentials["secret_key"]
    st.session_state.bucket_name = credentials["bucket_name"]

# Bouton de chargement des credentials
if st.sidebar.button("📂 Charger credentials depuis .env"):
    credentials = load_credentials()
    st.session_state.access_key = credentials["access_key"]
    st.session_state.secret_key = credentials["secret_key"]
    st.session_state.bucket_name = credentials["bucket_name"]
    st.sidebar.success("✅ Credentials chargés avec succès!")

# Champs pour les credentials AWS avec `st.session_state`
access_key = st.sidebar.text_input("Access Key", value=st.session_state.access_key, key="access_key", type="password")
secret_key = st.sidebar.text_input("Secret Key", value=st.session_state.secret_key, key="secret_key", type="password")
bucket_name = st.sidebar.text_input("Nom du bucket S3", value=st.session_state.bucket_name, key="bucket_name")

# Titre principal
st.markdown("""
# 📸 Content Moderator Pro
🔍 Analysez et modérez votre contenu en un clic!
""")

# Vérification des credentials AWS
if not st.session_state.access_key or not st.session_state.secret_key:
    st.warning("⚠️ Veuillez configurer vos credentials AWS dans la barre latérale")
else:
    st.success("✅ Credentials configurés, vous pouvez utiliser l'application!")
    
    # Initialisation du client S3
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=st.session_state.access_key,
        aws_secret_access_key=st.session_state.secret_key
    )
    
    # Zone d'upload de fichiers
    st.markdown("**Glissez-déposez un fichier (image ou vidéo) ici**")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "mp4", "avi", "mpeg4"], accept_multiple_files=False)
    
    if uploaded_file is not None:
        with st.spinner("🕒 Analyse en cours..."):
            # Vérification du format et de la taille
            allowed_types = {"image/jpeg", "image/png", "video/mp4", "video/avi", "video/mpeg4"}
            max_size = 50 * 1024 * 1024  # 50MB
            
            if uploaded_file.type not in allowed_types:
                st.error("❌ Format de fichier non supporté!")
            elif uploaded_file.size > max_size:
                st.error("❌ Fichier trop volumineux (max 50MB)!")
            else:
                original_extension = os.path.splitext(uploaded_file.name)[1]  # Ex: ".jpg", ".png", ".pdf"
    
                # Créer un fichier temporaire avec la même extension
                with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as temp_file:
                    temp_file.write(uploaded_file.getbuffer())  # Écrire le contenu du fichier uploadé
                    file_path = temp_file.name
                
                # Upload vers S3
                try:
                    # Analyse du fichier avec le module de modération
                    rekognition = boto3.client("rekognition", 
                                               aws_access_key_id=st.session_state.access_key, 
                                               aws_secret_access_key=st.session_state.secret_key)
                    
                    transcribe = boto3.client("transcribe", 
                                              aws_access_key_id=st.session_state.access_key, 
                                              aws_secret_access_key=st.session_state.secret_key)
                    
                    comprehend = boto3.client("comprehend", 
                                              aws_access_key_id=st.session_state.access_key, 
                                              aws_secret_access_key=st.session_state.secret_key)

                    analysis_results = process_media(file_path, rekognition, transcribe, comprehend,s3_client, st.session_state.bucket_name)

                    # Affichage sous forme de carte style réseau social
                    st.markdown("---")
                    st.subheader("📌 Contenu Uploadé")
                    if uploaded_file.type.startswith("image"):
                        st.image(uploaded_file, caption="Image uploadée", use_column_width=True)
                    elif uploaded_file.type.startswith("video"):
                        st.video(uploaded_file)
                    
                    # Génération des hashtags
                    hashtags = []
                    if analysis_results:
                        if 'objects' in analysis_results:
                            hashtags.extend(analysis_results['objects'])
                        if 'celebrities' in analysis_results:
                            hashtags.extend(analysis_results['celebrities'])
                        if 'emotions' in analysis_results:
                            hashtags.append(analysis_results['emotions'])

                    # Affichage des hashtags générés
                    st.markdown("### 🏷 Hashtags générés")
                    st.markdown(" ".join([f"#{tag}" for tag in hashtags]))

                    # Affichage des résultats de la modération
                    st.markdown("### 🚨 Résultats de la modération")
                    if analysis_results and 'moderation' in analysis_results and analysis_results['moderation']:
                        st.error("❌ Contenu inapproprié détecté!")
                        st.markdown("**Thèmes sensibles détectés :**")
                        for label in analysis_results['moderation']:
                            st.warning(f"⚠️ {label}")
                    else:
                        st.success("✅ Aucun contenu inapproprié détecté!")
                    
                    # Option de transcription pour les vidéos
                    if uploaded_file.type.startswith("video"):
                        st.markdown("### 🎤 Transcription de la vidéo")
                        st.text("[Texte généré automatiquement...] (Simulation)")
                
                except Exception as e:
                    st.error(f"Erreur lors de l'upload : {e}")
                finally:
                    os.remove(file_path)  # Nettoyage du fichier temporaire après upload
