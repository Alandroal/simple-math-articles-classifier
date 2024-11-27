import PyPDF2
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch
from pathlib import Path
import pandas as pd
from collections import Counter
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    logger.error(f"Error downloading NLTK data: {str(e)}")

def clean_text(text):
    """Clean text by removing stopwords, common words, and special characters."""
    # Get stop words
    stop_words = set(stopwords.words('english'))
    
    # Add common scientific paper words to stop words
    common_words = {
        'abstract', 'introduction', 'conclusion', 'references', 'et', 'al',
        'figure', 'table', 'section', 'results', 'discussion', 'method',
        'methods', 'analysis', 'study', 'studies', 'research', 'data',
        'using', 'used', 'based', 'paper', 'proposed', 'approach'
    }
    stop_words.update(common_words)
    
    # Convert to lowercase and tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation, numbers, and single characters
    tokens = [token for token in tokens 
             if token not in string.punctuation
             and not token.isnumeric()
             and len(token) > 1]
    
    # Remove stop words and common words
    tokens = [token for token in tokens if token not in stop_words]
    
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return None

def get_bert_embedding(text, model, tokenizer, max_length=512):
    """Generate BERT embedding for text."""
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                          truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        return embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        return None

def get_meaningful_words(text, top_n=5):
    """Extract most frequent meaningful words from text."""
    # Tokenize and count words
    words = word_tokenize(text.lower())
    
    # Filter out very short words and get word frequencies
    word_freq = Counter(word for word in words if len(word) > 2)
    
    # Get top N most common words
    return [word for word, _ in word_freq.most_common(top_n)]

def classify_papers(folder_path, num_clusters=5):
    """Classify papers using BERT embeddings and K-means clustering."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        raise ValueError(f"Folder not found: {folder_path}")
    
    # Load BERT model
    logger.info("Loading BERT model...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    # Process PDFs
    pdf_files = list(folder_path.glob('*.pdf'))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {folder_path}")
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    documents = []
    embeddings = []
    
    for pdf_path in pdf_files:
        logger.info(f"Processing {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        
        if text and len(text.strip()) > 0:
            embedding = get_bert_embedding(text, model, tokenizer)
            if embedding is not None:
                documents.append({
                    'filename': pdf_path.name,
                    'text': text[:2000],  # Store more text for better analysis
                })
                embeddings.append(embedding)
        else:
            logger.warning(f"Skipping {pdf_path.name} - no text extracted")
    
    if not embeddings:
        raise ValueError("No valid embeddings generated from PDF files")
    
    # Convert embeddings list to numpy array
    embeddings_array = np.vstack(embeddings)
    logger.info(f"Generated embeddings shape: {embeddings_array.shape}")
    
    # Adjust number of clusters if necessary
    num_clusters = min(num_clusters, len(embeddings))
    logger.info(f"Clustering with {num_clusters} clusters")
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=num_clusters)
    clusters = kmeans.fit_predict(embeddings_array)
    
    # Create summary
    results = pd.DataFrame(documents)
    results['cluster'] = clusters
    
    # Analyze clusters
    cluster_summaries = {}
    for cluster_id in range(num_clusters):
        cluster_docs = results[results['cluster'] == cluster_id]
        combined_text = ' '.join(cluster_docs['text'])
        key_terms = get_meaningful_words(combined_text)
        
        cluster_summaries[f'Cluster {cluster_id}'] = {
            'num_papers': len(cluster_docs),
            'key_terms': key_terms,
            'papers': cluster_docs['filename'].tolist()
        }
    
    return cluster_summaries

if __name__ == "__main__":
    try:
        folder_path = "/home/jraposo/Documents/Pessoal/Rumos/AI Course/Projecto/Data"  # Replace with your folder path
        logger.info(f"Starting classification for folder: {folder_path}")
        summaries = classify_papers(folder_path)
        
        # Print results
        for cluster, info in summaries.items():
            print(f"\n{cluster}:")
            print(f"Number of papers: {info['num_papers']}")
            print(f"Key terms: {', '.join(info['key_terms'])}")
            print("Papers:", ', '.join(info['papers']))
            
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")