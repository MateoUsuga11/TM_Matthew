import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image
from keras.models import load_model
import platform
import random

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes - Piedra, Papel o Tijera")

# Imagen de ejemplo en la app
image = Image.open('BannerImg.png')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine para jugar Piedra, Papel o Tijera")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # Preprocesamiento de la imagen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Predicción del modelo
    prediction = model.predict(data)
    print(prediction)

    # Mapeo de jugadas
    jugadas = ["Tijera", "Piedra", "Papel", "Nada"]
    jugador = None

    for i in range(4):
        if prediction[0][i] > 0.5:
            jugador = jugadas[i]
            st.header(f"Detectado: {jugador} (Probabilidad: {prediction[0][i]:.2f})")

    # Si el jugador hizo una jugada válida (no "Nada")
    if jugador in ["Tijera", "Piedra", "Papel"]:
        computadora = random.choice(["Tijera", "Piedra", "Papel"])
        st.subheader(f"La computadora eligió: {computadora}")

        # Mostrar imagen del robot según la elección de la computadora
        robot_images = {
            "Piedra": "RobotRock.png",
            "Papel": "RobotPaper.png",
            "Tijera": "RobotScissors.png"
        }

        if computadora in robot_images:
            robot_img = Image.open(robot_images[computadora])
            st.image(robot_img, caption=f"El robot eligió {computadora}", width=250)

        # Determinar el resultado
        if jugador == computadora:
            resultado = "Empate 🤝"
        elif (jugador == "Tijera" and computadora == "Papel") or \
             (jugador == "Piedra" and computadora == "Tijera") or \
             (jugador == "Papel" and computadora == "Piedra"):
            resultado = "¡Ganaste! 🎉"
        else:
            resultado = "Perdiste 😢"

        st.success(resultado)
    elif jugador == "Nada":
        st.warning("No se detectó una jugada válida, intenta de nuevo.")


