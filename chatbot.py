import streamlit as st
import speech_recognition as sr
import openai
import nltk
from nltk.chat.util import Chat, reflections

# Assurez-vous d'avoir téléchargé les ressources nltk nécessaires
nltk.download('punkt')

# Configurez votre clé API OpenAI
openai.api_key = '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'

# Préparez le chatbot avec des paires de phrases
pairs = [
    ['bonjour', 'salut, comment puis-je vous aider?'],
    ['comment ça va ?', 'Je vais bien, merci ! Et vous ?'],
    ['au revoir', 'À bientôt !'],
]

chatbot = Chat(pairs, reflections)


# Fonction pour obtenir une réponse d'OpenAI
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Utilisez le modèle de votre choix
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']


# Fonction de reconnaissance vocale
def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Veuillez parler...")
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio, language='fr-FR')
            st.write("Vous avez dit : " + text)
            return text
        except sr.UnknownValueError:
            st.error("Désolé, je n'ai pas compris.")
            return ""
        except sr.RequestError as e:
            st.error("Erreur de service; {0}".format(e))
            return ""


# Fonction pour obtenir la réponse du chatbot
def get_response(user_input):
    if user_input:
        # Utilisez OpenAI pour obtenir une réponse
        return get_openai_response(user_input)
    return "Je n'ai pas compris."


# Créer l'application Streamlit
def main():
    st.title("Chatbot à commande vocale avec OpenAI")

    user_input = st.text_input("Entrez votre message (ou utilisez la commande vocale)")

    if st.button("Utiliser la voix"):
        user_input = transcribe_audio()

    if user_input:
        response = get_response(user_input)
        st.write("Chatbot : " + response)


if __name__ == "__main__":
    main()