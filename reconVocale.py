import streamlit as st
import speech_recognition as sr
import os

# Fonction pour choisir l'API de reconnaissance vocale
def choose_recognition_api(api_name):
    if api_name == 'Google':
        return sr.Recognizer().recognize_google
    elif api_name == 'Sphinx':
        return sr.Recognizer().recognize_sphinx
    else:
        raise ValueError("API non reconnue.")

# Fonction pour transcrire la parole
def transcribe_speech(api, language):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Veuillez parler...")
        audio = recognizer.listen(source)
        try:
            text = api(audio, language=language)
            return text
        except sr.UnknownValueError:
            st.error("Désolé, je n'ai pas pu comprendre l'audio.")
        except sr.RequestError as e:
            st.error(f"Erreur de connexion avec l'API : {e}")

# Fonction pour enregistrer la transcription
def save_transcription(text, filename):
    with open(filename, 'w') as file:
        file.write(text)
    st.success(f"Transcription enregistrée dans {filename}")

# Application Streamlit
st.title("Application de Reconnaissance Vocale")

# Sélection de l'API
api_choice = st.selectbox("Choisissez l'API de reconnaissance vocale :", ["Google", "Sphinx"])

# Choix de la langue
language_choice = st.selectbox("Choisissez la langue :", ["fr-FR", "en-US"])

if st.button("Transcrire la parole"):
    api = choose_recognition_api(api_choice)
    text = transcribe_speech(api, language_choice)
    if text:
        st.write("Texte transcrit : ", text)
        if st.button("Enregistrer la transcription"):
            save_transcription(text, "transcription.txt")