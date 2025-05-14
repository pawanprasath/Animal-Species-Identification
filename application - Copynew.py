import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the model using tf.keras.models.load_model
model = tf.kermodel = tf.keras.models.load_model("hybrid_model_savedmodel")

print("Model loaded successfully.")

my_w = tk.Tk()
my_w.geometry("400x400")  # Size of the window
my_w.title('Classification System')
my_font1 = ('times', 18, 'bold')

filename = ""
uploaded_image_label = tk.Label(my_w)

l1 = tk.Label(my_w, text='Give Images', width=30, font=my_font1)
l1.grid(row=1, column=1)
b1 = tk.Button(my_w, text='Upload File', width=20, command=lambda: upload_file())
b1.grid(row=2, column=1, padx=5, pady=5)

b3 = tk.Button(my_w, text='Predict Output', width=20, command=lambda: predict())
b3.grid(row=6, column=1, padx=5, pady=5)


def upload_file():
    global filename
    f_types = [('ALL', '*')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    image = Image.open(filename)

    # Resize the image in the given (width, height)
    imgs = image.resize((224, 224))
    img = ImageTk.PhotoImage(imgs)
    uploaded_image_label.config(image=img)
    uploaded_image_label.image = img  # Keep a reference to avoid garbage collection
    uploaded_image_label.grid(row=9, column=1, padx=5, pady=5)
    print("Uploaded file:", filename)


def predict():
    global model
    global filename

    img = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)


    print("Input Image Array:")
    print(img_array)

    predictions = model.predict(img_array)
    print("Raw Predictions:")
    print(predictions)
    print("Predictions Probabilities:", predictions[0])


    predicted_class = np.argmax(predictions[0])
    print("Predicted Class:", predicted_class)

    disease_mapping = {
        0: "Bear",
	1: "Butterfly",
        2: "Bull",
	3: "Lion",
	4: "Tiger",
	5: "Panda",
	6: "Snake"
    }
    predicted_disease = disease_mapping.get(predicted_class, "Unknown")
    print("Predicted Disease:", predicted_disease)

    out = f"Result for the given Image: {predicted_disease}"
    print(out)

    from tkinter import messagebox

    my_w.geometry("400x400")  # Adjust the window size
    messagebox.showinfo("Result", out)
    print(" ")


my_w.mainloop()  # Keep the window open
