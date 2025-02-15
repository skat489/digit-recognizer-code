import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw
import joblib

class DigitRecognizer:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Digit Recognizer")
        self.window.geometry("600x400")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg='black')
        self.canvas.grid(row=0, column=0, padx=20, pady=20)
        
        # Frame for buttons
        button_frame = ttk.Frame(self.window)
        button_frame.grid(row=0, column=1, padx=20)
        
        # Prediction label
        self.prediction_label = ttk.Label(button_frame, text="Prediction: ", font=('Arial', 20))
        self.prediction_label.pack(pady=20)
        
        # Buttons
        predict_btn = ttk.Button(button_frame, text="Predict", command=self.predict)
        predict_btn.pack(pady=5)
        
        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(pady=5)
        
        # Drawing setup
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_coordinates)
        
        self.old_x = None
        self.old_y = None
        
        # Load and train the model
        self.load_and_train_model()
        
    def paint(self, event):
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, event.x, event.y, 
                                  width=20, fill='white', capstyle=tk.ROUND, 
                                  smooth=tk.TRUE)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], 
                         fill='white', width=20)
        self.old_x = event.x
        self.old_y = event.y
        
    def reset_coordinates(self, event):
        self.old_x = None
        self.old_y = None
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new('L', (280, 280), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Prediction: ")
        
    def load_and_train_model(self):
        print("Loading and preprocessing data...")
        # Load training data
        train_data = pd.read_csv('train.csv')
        X_train = train_data.drop('label', axis=1).values
        y_train = train_data['label'].values
        
        # Scale the data
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create and train the model
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=50,
            random_state=42,
            verbose=True
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        print("Model training completed!")
        
    def preprocess_image(self):
        # Resize image to 28x28
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        # Convert to numpy array and normalize
        img_array = np.array(img_resized)
        # Flatten the image
        img_array = img_array.reshape(1, -1)
        # Scale the image
        img_array_scaled = self.scaler.transform(img_array)
        return img_array_scaled
        
    def predict(self):
        # Preprocess the drawn image
        img_array_scaled = self.preprocess_image()
        # Make prediction
        prediction = self.model.predict(img_array_scaled)
        # Update label
        self.prediction_label.config(text=f"Prediction: {prediction[0]}")
        
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = DigitRecognizer()
    app.run()
