# import mlflow.pyfunc
# import torch
# from torchvision import transforms
# from PIL import Image
# import os
# import numpy as np

# # # Load model from MLflow Model Registry
# # MODEL_NAME = "CatDogModel"
# # STAGE = "Production"  # or use version number like "models:/CatDogModel/7"
# # model_uri = f"models:/{MODEL_NAME}/{STAGE}"

# model_uri = "models:/CatDogModel/7"


# print(f"Loading model from: {model_uri}")
# model = mlflow.pyfunc.load_model(model_uri)

# # Define the same preprocessing used during training
# preprocess = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# def load_and_preprocess_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     img = preprocess(img)
#     img = img.unsqueeze(0)  # Add batch dimension
#     return img

# # Load a sample image for testing
# sample_path = "dataset/test_set/dogs/dog.4045.jpg"   

# input_tensor = load_and_preprocess_image(sample_path)

# # Predict (note: PyFunc model expects numpy input)
# input_numpy = input_tensor.numpy()
# predictions = model.predict(input_numpy)

# # Convert sigmoid output to binary class
# predicted_class = int((predictions[0] > 0.5).item())
# label_map = {0: "Cat", 1: "Dog"}
# print(f"Prediction: {label_map[predicted_class]} (Raw: {predictions[0].item():.4f})")







import mlflow.pyfunc
from torchvision import transforms
from PIL import Image
import numpy as np
import time
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

model_uri = "models:/CatDogModel/7"

def load_model_with_spinner(uri):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Loading MLflow model...", start=False)
        progress.start_task(task)
        # simulate loading progress for visual effect (you can remove time.sleep)
        time.sleep(1)
        model = mlflow.pyfunc.load_model(uri)
        time.sleep(0.5)
    return model

console.print(f"[bold blue]Model URI:[/bold blue] {model_uri}")
model = load_model_with_spinner(model_uri)
console.print("[green]Model loaded successfully![/green]\n")

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_and_preprocess_image(image_path):
    console.print(f"[yellow]Loading image:[/yellow] {image_path}")
    img = Image.open(image_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

sample_path = "dataset/test_set/dogs/dog.4045.jpg"
input_tensor = load_and_preprocess_image(sample_path)

console.print("[blue]Running prediction...[/blue]")
input_numpy = input_tensor.numpy()
predictions = model.predict(input_numpy)

predicted_class = int((predictions[0] > 0.5).item())
label_map = {0: "Cat", 1: "Dog"}
confidence = predictions[0].item()

console.print(f"\n[bold green]Prediction:[/bold green] [bold magenta]{label_map[predicted_class]}[/bold magenta]")
console.print(f"[bold cyan]Confidence score:[/bold cyan] {confidence:.4f}")
