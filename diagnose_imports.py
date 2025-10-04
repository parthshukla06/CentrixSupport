modules = [
    'groq', 'flask', 'numpy', 'torch', 'numba', 'llvmlite', 'onnxruntime', 'whisper',
    'langchain', 'chromadb', 'pdfplumber', 'pytesseract', 'cv2', 'pydub', 'playsound', 'nltk'
]

for m in modules:
    try:
        __import__(m)
        print(f'OK: {m}')
    except Exception as e:
        print(f'FAILED: {m} -> {type(e).__name__}: {e}')
        break
