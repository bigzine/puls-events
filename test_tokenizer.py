import time
print("Début du téléchargement du tokenizer Mixtral...")
start = time.time()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
print(f"Terminé en {time.time() - start:.1f}s")