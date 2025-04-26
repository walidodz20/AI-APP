import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from huggingface_hub import InferenceClient
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import io
from io import BytesIO
import requests

# Remplace ce token par le tien de manière sécurisée (variable d'environnement recommandée en production)
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

# Initialisation de l'app FastAPI
app = FastAPI()

# Autoriser les requêtes Cross-Origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialisation des clients Hugging Face avec authentification
summary_client = InferenceClient(model="facebook/bart-large-cnn", token=HUGGINGFACE_TOKEN)
qa_client = InferenceClient(model="deepset/roberta-base-squad2", token=HUGGINGFACE_TOKEN)
image_caption_client = InferenceClient(model="nlpconnect/vit-gpt2-image-captioning", token=HUGGINGFACE_TOKEN)

# Extraction du texte des fichiers
def extract_text_from_pdf(content: bytes) -> str:
    text = ""
    reader = PdfReader(io.BytesIO(content))
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(content: bytes) -> str:
    text = ""
    doc = Document(io.BytesIO(content))
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.strip()

def process_uploaded_file(file: UploadFile) -> str:
    content = file.file.read()
    extension = file.filename.split('.')[-1].lower()

    if extension == "pdf":
        return extract_text_from_pdf(content)
    elif extension == "docx":
        return extract_text_from_docx(content)
    elif extension == "txt":
        return content.decode("utf-8").strip()
    else:
        raise ValueError("Type de fichier non supporté")

# Point d'entrée HTML
@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Résumé
@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    try:
        text = process_uploaded_file(file)

        if len(text) < 20:
            return {"summary": "Document trop court pour être résumé."}

        summary = summary_client.summarization(text[:3000])
        return {"summary": summary}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur lors de l'analyse: {str(e)}"})

# Question-Réponse
@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        # Determine if the file is an image
        content_type = file.content_type
        if content_type.startswith("image/"):
            image_bytes = await file.read()
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_pil.thumbnail((1024, 1024))

            img_byte_arr = BytesIO()
            image_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Generate image description
            result = image_caption_client.image_to_text(img_byte_arr)
            if isinstance(result, dict):
                context = result.get("generated_text") or result.get("caption") or ""
            elif isinstance(result, list) and len(result) > 0:
                context = result[0].get("generated_text", "")
            elif isinstance(result, str):
                context = result
            else:
                context = ""

        else:
            # Not an image, process as document
            text = process_uploaded_file(file)
            if len(text) < 20:
                return {"answer": "Document trop court pour répondre à la question."}
            context = text[:3000]

        if not context:
            return {"answer": "Aucune information disponible pour répondre à la question."}

        result = qa_client.question_answering(question=question, context=context)
        return {"answer": result.get("answer", "Aucune réponse trouvée.")}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur lors de la recherche de réponse: {str(e)}"})
        
# Interprétation d'Image
@app.post("/interpret_image")
async def interpret_image(image: UploadFile = File(...)):
    try:
        # Lire l'image
        image_bytes = await image.read()

        # Ouvrir l'image avec PIL
        image_pil = Image.open(io.BytesIO(image_bytes))
        image_pil = image_pil.convert("RGB")
        image_pil.thumbnail((1024, 1024))

        # Convertir en bytes (JPEG)
        img_byte_arr = BytesIO()
        image_pil.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Appeler le modèle
        result = image_caption_client.image_to_text(img_byte_arr)

        # 🔍 Affichage du résultat brut pour débogage
        print("Résultat brut du modèle image-to-text:", result)

        # Extraire la description si disponible
        if isinstance(result, dict):
            description = result.get("generated_text") or result.get("caption") or "Description non trouvée."
        elif isinstance(result, list) and len(result) > 0:
            description = result[0].get("generated_text", "Description non trouvée.")
        elif isinstance(result, str):
            description = result
        else:
            description = "Description non trouvée."

        return {"description": description}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Erreur lors de l'interprétation de l'image: {str(e)}"})

# Démarrage local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
