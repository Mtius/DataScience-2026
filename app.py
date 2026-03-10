import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="IA para leer números", layout="centered")

@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("modelo_entrenado.keras", compile=False)

model = cargar_modelo()

st.title("Inteligencia Artificial para leer números")
st.subheader("Dibuja un dígito (0-9) con lápiz blanco sobre fondo negro. La imagen se redimensiona a 28x28 y se normaliza para la predicción del modelo Keras/TensorFlow.")

st.write("Grosor del lápiz: 4 — Fondo: negro — Lápiz: blanco")

canvas_result = st_canvas(
    fill_color="rgba(0,0,0,1)",
    stroke_width=4,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Capturar y predecir"):
    if canvas_result.image_data is None:
        st.error("No hay imagen dibujada en el canvas.")
    else:
        img_data = canvas_result.image_data.astype('uint8')
        img = Image.fromarray(img_data, 'RGBA').convert('L')
        try:
            resample = Image.Resampling.LANCZOS
        except Exception:
            resample = Image.ANTIALIAS
        img_resized = img.resize((28, 28), resample)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        preds = model.predict(arr)
        pred_class = int(np.argmax(preds[0]))
        pred_prob = float(preds[0][pred_class])

        st.success(f"Predicción: {pred_class}")
        st.write(f"Probabilidad: {pred_prob*100:.2f}%")

        # Mostrar la imagen procesada (escalada para visualizar)
        st.image(img_resized.resize((140, 140)), caption="Imagen procesada (28x28 escalada)")

st.markdown("---")
st.write("© Derechos registrados Unab 2026 — realizado por: Santiago Matheus 😊")
