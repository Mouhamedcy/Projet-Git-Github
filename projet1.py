import streamlit as st
import requests
import base64
import json

import os
from io import BytesIO
import cv2
import numpy as np
from PIL import Image

# Configuration de l'API Claude
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_KEY = "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"


def get_image_description(image_data):
    """Envoie l'image à l'API Claude pour obtenir une description."""
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Encodage de l'image en base64
    base64_image = base64.b64encode(image_data).decode('utf-8')

    # Construction du message selon le format de l'API
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": "Décrivez cette image en détail, s'il vous plaît."
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(CLAUDE_API_URL, json=data, headers=headers)
        response.raise_for_status()

        result = response.json()
        if "content" in result and len(result["content"]) > 0:
            for content in result["content"]:
                if content["type"] == "text":
                    return content["text"]
        return "Aucune description trouvée dans la réponse."
    except requests.exceptions.RequestException as e:
        return f"Erreur lors de la requête API: {str(e)}"
    except json.JSONDecodeError:
        return "Erreur: Réponse non-JSON reçue de l'API."
    except Exception as e:
        return f"Erreur inattendue: {str(e)}"


def compare_images_with_claude(img1_data, img2_data):
    """Envoie les deux images à l'API Claude pour obtenir une comparaison."""
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    # Encodage des images en base64
    base64_img1 = base64.b64encode(img1_data).decode('utf-8')
    base64_img2 = base64.b64encode(img2_data).decode('utf-8')

    # Construction du message selon le format de l'API
    data = {
        "model": "claude-3-7-sonnet-20250219",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Voici deux images. Veuillez comparer ces images et décrire les différences et similitudes entre elles."
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_img1
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_img2
                        }
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(CLAUDE_API_URL, json=data, headers=headers)
        response.raise_for_status()

        result = response.json()
        if "content" in result and len(result["content"]) > 0:
            for content in result["content"]:
                if content["type"] == "text":
                    return content["text"]
        return "Aucune comparaison trouvée dans la réponse."
    except requests.exceptions.RequestException as e:
        return f"Erreur lors de la requête API: {str(e)}"
    except json.JSONDecodeError:
        return "Erreur: Réponse non-JSON reçue de l'API."
    except Exception as e:
        return f"Erreur inattendue: {str(e)}"


def text_to_speech(text, lang='fr'):
    """Convertit le texte en audio et retourne le fichier audio."""
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes


def compare_images(img1_data, img2_data):
    """Compare deux images et retourne une image avec les différences surlignées."""
    # Convertir les données binaires en images OpenCV
    img1 = Image.open(BytesIO(img1_data))
    img2 = Image.open(BytesIO(img2_data))

    # Redimensionner les images à la même taille si nécessaire
    img2 = img2.resize(img1.size)

    # Convertir en numpy arrays pour OpenCV
    img1_cv = np.array(img1)
    img2_cv = np.array(img2)

    # Convertir en gris pour simplifier la comparaison
    if len(img1_cv.shape) == 3:  # Image couleur
        img1_gray = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2_cv, cv2.COLOR_RGB2GRAY)
    else:  # Image déjà en noir et blanc
        img1_gray = img1_cv
        img2_gray = img2_cv

    # Calculer la différence absolue entre les images
    diff = cv2.absdiff(img1_gray, img2_gray)

    # Appliquer un seuil pour mettre en évidence les différences
    thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilater pour mieux voir les différences
    kernel = np.ones((5, 5), np.uint8)
    dilated_diff = cv2.dilate(thresh, kernel, iterations=1)

    # Créer une image en couleur pour afficher les différences
    diff_color = cv2.cvtColor(img1_cv, cv2.COLOR_RGB2BGR) if len(img1_cv.shape) == 3 else cv2.cvtColor(img1_gray,
                                                                                                       cv2.COLOR_GRAY2BGR)

    # Marquer les différences en rouge
    diff_color[dilated_diff > 0] = [0, 0, 255]  # BGR: Rouge

    # Convertir en RGB pour Streamlit
    diff_color_rgb = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

    # Calculer le pourcentage de différence
    diff_percentage = (np.count_nonzero(dilated_diff) / dilated_diff.size) * 100

    return diff_color_rgb, diff_percentage


# Interface Streamlit
st.title("Analyse d'image avec Claude")

# Navigation entre les fonctionnalités
app_mode = st.sidebar.selectbox("Choisissez une fonctionnalité",
                                ["Description d'image", "Comparaison d'images"])

if app_mode == "Description d'image":
    st.header("Description d'image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "png", "jpeg"], key="single_img")

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        st.image(image_data, caption="Image chargée", use_container_width=True)

        if st.button("Générer une description"):
            with st.spinner("Analyse en cours..."):
                description = get_image_description(image_data)
                st.write("### Description générée:")
                st.write(description)

                # Génération de l'audio
                st.write("### Écouter la description:")
                with st.spinner("Génération de l'audio..."):
                    audio_file = text_to_speech(description)
                    st.audio(audio_file, format='audio/mp3')

                    st.download_button(
                        label="Télécharger l'audio",
                        data=audio_file,
                        file_name="description_audio.mp3",
                        mime="audio/mp3"
                    )

elif app_mode == "Comparaison d'images":
    st.header("Comparaison d'images")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Première image")
        img1_file = st.file_uploader("Choisissez la première image", type=["jpg", "png", "jpeg"], key="img1")
        if img1_file is not None:
            img1_data = img1_file.read()
            st.image(img1_data, caption="Image 1", use_container_width=True)

            # Bouton pour décrire la première image
            if st.button("Décrire l'image 1"):
                with st.spinner("Analyse de l'image 1 en cours..."):
                    img1_description = get_image_description(img1_data)
                    st.write("#### Description de l'image 1:")
                    st.write(img1_description)

                    # Audio pour l'image 1
                    audio_img1 = text_to_speech(img1_description)
                    st.audio(audio_img1, format='audio/mp3')

    with col2:
        st.write("Deuxième image")
        img2_file = st.file_uploader("Choisissez la deuxième image", type=["jpg", "png", "jpeg"], key="img2")
        if img2_file is not None:
            img2_data = img2_file.read()
            st.image(img2_data, caption="Image 2", use_container_width=True)

            # Bouton pour décrire la deuxième image
            if st.button("Décrire l'image 2"):
                with st.spinner("Analyse de l'image 2 en cours..."):
                    img2_description = get_image_description(img2_data)
                    st.write("#### Description de l'image 2:")
                    st.write(img2_description)

                    # Audio pour l'image 2
                    audio_img2 = text_to_speech(img2_description)
                    st.audio(audio_img2, format='audio/mp3')

    if img1_file is not None and img2_file is not None:
        # Séparation visuelle
        st.markdown("---")
        st.subheader("Options de comparaison")

        # Bouton pour comparer visuellement
        compare_btn = st.button("Comparer visuellement les images")
        if compare_btn:
            with st.spinner("Comparaison visuelle en cours..."):
                diff_image, diff_percentage = compare_images(img1_data, img2_data)

                st.write(f"### Différence détectée: {diff_percentage:.2f}%")
                st.image(diff_image, caption="Différences (en rouge)", use_container_width=True)

                # Créer l'image de différence au format bytes pour Claude
                diff_pil = Image.fromarray(diff_image)
                diff_bytes = BytesIO()
                diff_pil.save(diff_bytes, format="JPEG")
                diff_bytes.seek(0)

                # Stocker l'image de différence dans session_state pour l'utiliser avec le second bouton
                st.session_state.diff_bytes = diff_bytes.getvalue()

        # Bouton pour analyser l'image de différence
        if 'diff_bytes' in st.session_state and st.button("Analyser l'image de différence"):
            st.write("### Analyse de l'image de différence")
            with st.spinner("Analyse de l'image de différence en cours..."):
                diff_description = get_image_description(st.session_state.diff_bytes)
                st.write(diff_description)

                # Option audio pour la description des différences
                with st.spinner("Génération de l'audio..."):
                    diff_audio = text_to_speech(diff_description)
                    st.audio(diff_audio, format='audio/mp3')

        # Bouton pour comparer directement les deux images avec Claude
        st.markdown("---")
        st.subheader("Comparaison intelligente avec Claude")
        if st.button("Comparer directement les deux images avec Claude"):
            with st.spinner("Claude analyse les deux images..."):
                comparison_description = compare_images_with_claude(img1_data, img2_data)
                st.write("### Comparaison générée par Claude:")
                st.write(comparison_description)

                # Option audio pour la comparaison
                with st.spinner("Génération de l'audio..."):
                    comparison_audio = text_to_speech(comparison_description)
                    st.audio(comparison_audio, format='audio/mp3')
