import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import tensorflow as tf
import numpy as np

# Taille de la grille et de l'image
CANVAS_SIZE = 280
IMAGE_SIZE = 28

# Charger le modèle
model = tf.keras.models.load_model('number_reco/model.h5')

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dessinez un chiffre")
        
        self.canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='white')
        self.canvas.pack()
        
        self.canvas.bind("<B1-Motion>", self.draw)
        
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color='white')
        self.draw = ImageDraw.Draw(self.image)
        
        self.reset_button = tk.Button(root, text="Effacer", command=self.reset)
        self.reset_button.pack(side=tk.BOTTOM)
        
        self.predict_button = tk.Button(root, text="Prédire", command=self.predict)
        self.predict_button.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def reset(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color='white')
        self.draw = ImageDraw.Draw(self.image)

    def predict(self):
        # Prétraitement de l'image
        resized_image = self.image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        inverted_image = ImageOps.invert(resized_image)
        image_array = np.array(inverted_image)
        image_array = image_array.astype('float32') / 255.0
        image_array = image_array.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
        
        # Faire la prédiction avec le modèle
        prediction = model.predict(image_array)
        predicted_digit = np.argmax(prediction)
        
        # Afficher la prédiction
        prediction_text = f"Chiffre prédit : {predicted_digit}"
        self.root.title(prediction_text)

# Création de l'interface graphique
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()
