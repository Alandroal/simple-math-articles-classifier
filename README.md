# simple-math-articles-classifier

 Math PDF files are located in folder Data

There are two versions of the classifier

Version V2 (without OCR)
 
  Classifier: K-Means
  Embedding: BERT
  Dependencies: pip install PyPDF2 numpy transformers scikit-learn torch pandas collections logging nltk

Version V3 (with OCR and visualisation) [needs as input in th code a pre determined number of clusters]  
 
  Classifier: DOC2VEC
  Embedding: SciBERT
  Dependencies: pip install pdfplumber numpy matplotlib seaborn pillow pytesseract spacy gensim transformers torch scikit-learn logging os pandas pdf2image  
