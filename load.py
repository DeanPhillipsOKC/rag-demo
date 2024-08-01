import os
import shutil
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Define the folder paths
input_folder = "intake"
processed_folder = os.path.join(input_folder, "processed")

# Create the processed folder if it doesn't exist
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

# Initialize the embeddings model (you can use different models if desired)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_db = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Loop through all files in the intake folder
for filename in os.listdir(input_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, filename)
        
        # Open and read the PDF file
        with open(pdf_path, "rb") as pdf_file:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

            # Embed the text using LangChain and store it in ChromaDB
            document_id = filename.split(".")[0]
            chroma_db.add_texts([text], ids=[document_id])
            chroma_db.persist()

        # Move the processed PDF to the processed folder
        shutil.move(pdf_path, os.path.join(processed_folder, filename))

print("All PDFs processed, embedded, and moved to the 'intake/processed' folder.")
