import pdfplumber
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract
import spacy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import defaultdict
import logging
import os
from pathlib import Path
import pandas as pd
from pdf2image import convert_from_path

class MathDocumentClassifier:
    def __init__(self, vector_size=100, min_count=2, epochs=40):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.nlp = self._init_spacy()
        self.tokenizer, self.model = self._init_scibert()
        
        # Doc2Vec parameters
        self.vector_size = vector_size
        self.min_count = min_count
        self.epochs = epochs
        self.doc2vec_model = None
        
        # Math-specific keywords for enhanced classification
        self.math_keywords = {
            'algebra': ['equation', 'polynomial', 'variable', 'matrix', 'vector'],
            'calculus': ['derivative', 'integral', 'limit', 'differential', 'continuous'],
            'geometry': ['angle', 'triangle', 'circle', 'polygon', 'coordinate'],
            'statistics': ['probability', 'distribution', 'random', 'variance', 'mean']
        }
        
        # Results storage
        self.results = defaultdict(dict)
        self.documents = []
        self.document_labels = []

    def _init_spacy(self):
        """Initialize SpaCy model"""
        try:
            return spacy.load('en_core_web_sm')
        except OSError:
            self.logger.error("SpaCy model not found. Installing...")
            os.system('python -m spacy download en_core_web_sm')
            return spacy.load('en_core_web_sm')

    def _init_scibert(self):
        """Initialize SciBERT model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            return tokenizer, model
        except Exception as e:
            self.logger.error(f"Error initializing SciBERT: {str(e)}")
            raise

    def extract_text(self, pdf_path):
        """Extract text from PDF using PDFPlumber or Pytesseract"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    extracted = page.extract_text() or ""
                    text += extracted
                
                if not text.strip():
                    return self._ocr_process(pdf_path)
                return text
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}")
            return None

    def _ocr_process(self, pdf_path):
        """Process scanned PDF using Pytesseract"""
        try:
            text = ""
            images = convert_from_path(pdf_path)
            for image in images:
                text += pytesseract.image_to_string(image) + "\n"
            return text
        except Exception as e:
            self.logger.error(f"OCR processing error: {str(e)}")
            return None

    def preprocess_text(self, text):
        """Preprocess text with focus on mathematical content"""
        doc = self.nlp(text)
        
        # Basic preprocessing
        tokens = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.is_alpha:
                # Keep mathematical symbols and terms
                if (token.like_num or 
                    token.text in "+-*/=∫∑∏" or 
                    any(token.text.lower() in keywords 
                        for keywords in self.math_keywords.values())):
                    tokens.append(token.text.lower())
                else:
                    tokens.append(token.lemma_.lower())
        
        return tokens

    def embed_document(self, text):
        """Generate SciBERT embeddings"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()

    def train_doc2vec(self):
        """Train Doc2Vec model"""
        tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(self.documents)]
        self.doc2vec_model = Doc2Vec(
            vector_size=self.vector_size,
            min_count=self.min_count,
            epochs=self.epochs
        )
        self.doc2vec_model.build_vocab(tagged_docs)
        self.doc2vec_model.train(tagged_docs, total_examples=self.doc2vec_model.corpus_count, epochs=self.epochs)

    def cluster_documents(self, n_clusters=4):
        """Cluster documents using KMeans"""
        vectors = [self.doc2vec_model.dv[i] for i in range(len(self.documents))]
        
        # Adjust number of clusters if needed
        n_clusters = min(n_clusters, len(vectors))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(vectors), vectors

    def analyze_clusters(self, clusters):
        """Analyze cluster contents for mathematical topics"""
        cluster_keywords = defaultdict(list)
        for idx, cluster_id in enumerate(clusters):
            doc_tokens = self.documents[idx]
            for category, keywords in self.math_keywords.items():
                if any(keyword in doc_tokens for keyword in keywords):
                    cluster_keywords[cluster_id].append(category)
        
        # Determine dominant topic per cluster
        cluster_topics = {}
        for cluster_id, topics in cluster_keywords.items():
            if topics:
                most_common = max(set(topics), key=topics.count)
                cluster_topics[cluster_id] = most_common
            else:
                cluster_topics[cluster_id] = 'miscellaneous'
        
        return cluster_topics

    def process_folder(self, folder_path, n_clusters=4):
        """Process all PDFs in folder"""
        pdf_files = list(Path(folder_path).glob('*.pdf'))
        
        if not pdf_files:
            raise ValueError(f"No PDF files found in {folder_path}")
        
        for pdf_file in pdf_files:
            self.logger.info(f"Processing {pdf_file.name}")
            
            # Extract and process text
            text = self.extract_text(pdf_file)
            if not text:
                continue
            
            tokens = self.preprocess_text(text)
            self.documents.append(tokens)
            self.document_labels.append(pdf_file.name)
            
            # Generate embeddings
            self.results[pdf_file.name]['embeddings'] = self.embed_document(text)
        
        if not self.documents:
            raise ValueError("No documents were successfully processed")
        
        # Train model and cluster
        self.train_doc2vec()
        clusters, vectors = self.cluster_documents(n_clusters)
        cluster_topics = self.analyze_clusters(clusters)
        
        # Store results
        for idx, label in enumerate(self.document_labels):
            self.results[label].update({
                'cluster': int(clusters[idx]),
                'topic': cluster_topics[clusters[idx]],
                'vector': vectors[idx]
            })

    def visualize_results(self, output_dir='math_classification_results'):
        """Create visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Topic Distribution
        plt.figure(figsize=(12, 6))
        topics = [result['topic'] for result in self.results.values()]
        sns.countplot(y=topics)
        plt.title('Distribution of Mathematical Topics')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_distribution.png')
        plt.close()
        
        # 2. Document Similarity Heatmap
        similarity_matrix = np.zeros((len(self.documents), len(self.documents)))
        for i in range(len(self.documents)):
            for j in range(len(self.documents)):
                similarity_matrix[i, j] = np.dot(
                    self.doc2vec_model.dv[i],
                    self.doc2vec_model.dv[j]
                )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, xticklabels=self.document_labels, 
                   yticklabels=self.document_labels, cmap='coolwarm')
        plt.title('Document Similarity Matrix')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_matrix.png')
        plt.close()
        
        # 3. t-SNE Visualization with adjusted perplexity
        vectors = np.array([result['vector'] for result in self.results.values()])
        n_samples = len(vectors)
        
        if n_samples > 1:
            # Adjust perplexity based on number of samples
            perplexity = min(30, n_samples - 1)
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                vectors_2d = tsne.fit_transform(vectors)
                
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                                   c=[result['cluster'] for result in self.results.values()],
                                   cmap='viridis')
                plt.colorbar(scatter)
                plt.title('Document Clustering Visualization')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/document_clusters.png')
                plt.close()
            except Exception as e:
                self.logger.warning(f"Could not generate t-SNE visualization: {str(e)}")
        
        # 4. Generate Summary Report
        summary = pd.DataFrame([{
            'Document': name,
            'Topic': data['topic'],
            'Cluster': data['cluster']
        } for name, data in self.results.items()])
        
        summary.to_csv(f'{output_dir}/classification_summary.csv', index=False)
        
        # Print summary to console
        print("\nDocument Classification Summary:")
        print(summary.to_string())
        
        return summary

# Example usage
if __name__ == "__main__":
    try:
        classifier = MathDocumentClassifier()
        classifier.process_folder("/home/jraposo/Documents/Pessoal/Rumos/AI Course/Projecto/Data")
        summary = classifier.visualize_results()
        print("\nClassification completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")