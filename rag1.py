import os
import sys
import threading
import numpy as np
import pickle
from pathlib import Path
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                           QLabel, QPushButton, QTextEdit, QLineEdit, QFileDialog, 
                           QTabWidget, QListWidget, QListWidgetItem, QSplitter, QProgressBar,
                           QMessageBox, QFrame, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QPixmap, QFontMetrics


class SimpleRAGModel:
    def __init__(self, api_key=None):
        # Set API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required!")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-1.5-pro-latest")
        
        # Document storage and vectorization
        self.documents = []
        self.doc_ids = []
        self.doc_sources = []
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.document_vectors = None
        self.is_vectorizer_fitted = False
        
        # Cache for embeddings
        self.cache_file = Path("document_cache.pkl")
        self.load_cache()
        
        # For thread safety
        self.lock = threading.Lock()
        
    def load_cache(self):
        """Load document cache if it exists"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.documents = cache_data.get('documents', [])
                    self.doc_ids = cache_data.get('doc_ids', [])
                    self.doc_sources = cache_data.get('doc_sources', [])
                    self.vectorizer = cache_data.get('vectorizer', TfidfVectorizer(stop_words='english'))
                    self.document_vectors = cache_data.get('vectors', None)
                    self.is_vectorizer_fitted = cache_data.get('is_fitted', False)
                print(f"Loaded {len(self.documents)} documents from cache")
            except Exception as e:
                print(f"Failed to load cache: {e}")
    
    def save_cache(self):
        """Save document cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'doc_ids': self.doc_ids,
                'doc_sources': self.doc_sources,
                'vectorizer': self.vectorizer,
                'vectors': self.document_vectors,
                'is_fitted': self.is_vectorizer_fitted
            }, f)
    
    def add_documents(self, docs, sources=None, ids=None):
        """Add documents to the knowledge base"""
        if not docs:
            return []
            
        if ids is None:
            start_id = len(self.documents)
            ids = [f"doc_{i}" for i in range(start_id, start_id + len(docs))]
        
        if sources is None:
            sources = ["manual_entry" for _ in range(len(docs))]
        
        with self.lock:
            self.documents.extend(docs)
            self.doc_ids.extend(ids)
            self.doc_sources.extend(sources)
            
            # Rebuild the vector representation
            if not self.documents:
                return ids
                
            # Check if we're adding the first documents
            if not self.is_vectorizer_fitted:
                self.document_vectors = self.vectorizer.fit_transform(self.documents)
                self.is_vectorizer_fitted = True
            else:
                # For subsequent documents, we need to use the same vocabulary
                # So we transform all documents again
                self.document_vectors = self.vectorizer.transform(self.documents)
            
            self.save_cache()
        
        return ids
    
    def add_document_from_file(self, file_path):
        """Add content from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            doc_id = f"file_{Path(file_path).stem}"
            return self.add_documents([content], [str(file_path)], [doc_id])
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
    
    def retrieve_context(self, query, top_k=3):
        """Retrieve the most relevant documents for a query"""
        if not self.documents or not self.is_vectorizer_fitted:
            return []
        
        with self.lock:
            try:
                # Create query vector
                query_vector = self.vectorizer.transform([query])
                
                # Calculate similarity scores
                similarity_scores = cosine_similarity(query_vector, self.document_vectors).flatten()
                
                # Get top-k documents
                top_indices = similarity_scores.argsort()[-top_k:][::-1]
                
                # Filter for documents with some relevance
                relevant_docs = [(self.documents[i], self.doc_sources[i], similarity_scores[i]) 
                                for i in top_indices if similarity_scores[i] > 0.1]
                
                return relevant_docs
            except Exception as e:
                print(f"Error in retrieve_context: {e}")
                return []
    
    def get_answer(self, question, temperature=0.7):
        """Generate an answer based on retrieved context"""
        # Retrieve relevant context
        context_with_scores = self.retrieve_context(question)
        
        if not context_with_scores:
            # Fall back to no-context response
            return self.model.generate_content(
                f"Answer this question concisely: {question}",
                generation_config={"temperature": temperature}
            ).text, []
        
        # Build context string, ordered by relevance
        context_str = "\n\n".join([f"Document (source: {source}, relevance: {score:.2f}):\n{doc}" 
                                  for doc, source, score in context_with_scores])
        
        # Build prompt with context
        prompt = f"""Please answer the following question based ONLY on the provided context.
If the context doesn't contain enough information, say so rather than making up an answer.

Question: {question}

Context:
{context_str}

Answer:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"temperature": temperature}
            )
            return response.text, context_with_scores
        except Exception as e:
            return f"Error generating response: {str(e)}", context_with_scores

    def initialize_with_samples(self):
        """Add some sample documents to get started"""
        samples = [
            "RAG (Retrieval-Augmented Generation) is an AI framework that combines search capabilities with generative AI. It retrieves relevant information from a knowledge base before generating a response.",
            "Security threats often include phishing, malware, and ransomware. Phishing attempts to steal sensitive information by impersonating trusted entities.",
            "Data privacy regulations like GDPR and CCPA require companies to protect user data and provide transparency about data collection and usage.",
            "Zero-trust security is a framework requiring all users to be authenticated and authorized continuously, even those inside the network perimeter.",
            "Encryption is the process of encoding information to prevent unauthorized access. Common types include symmetric and asymmetric encryption."
        ]
        
        sources = [
            "AI Security Documentation",
            "Threat Intelligence Report",
            "Privacy Regulations Overview",
            "Security Architecture Guide",
            "Encryption Best Practices"
        ]
        
        self.add_documents(samples, sources)
        return len(samples)


class WorkerThread(QThread):
    update_signal = pyqtSignal(str, list)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    
    def __init__(self, rag_model, question, temperature=0.7):
        super().__init__()
        self.rag_model = rag_model
        self.question = question
        self.temperature = temperature
        
    def run(self):
        try:
            self.progress_signal.emit(30)  # Start progress
            answer, contexts = self.rag_model.get_answer(self.question, self.temperature)
            self.progress_signal.emit(100)  # Complete progress
            self.update_signal.emit(answer, contexts)
        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
            self.progress_signal.emit(0)  # Reset progress


class DarkFrame(QFrame):
    """Custom QFrame with Windows 11-style dark theme"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            DarkFrame {
                background-color: #1f1f1f;
                border-radius: 10px;
                border: 1px solid #333333;
            }
        """)


class RAGSecurityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rag_model = None
        self.api_key = os.environ.get("GOOGLE_API_KEY", "")
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("RAG Security Center")
        self.setMinimumSize(1000, 700)
        
        # Set the dark theme
        self.setup_dark_theme()
        
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Header with logo and title
        header_frame = DarkFrame()
        header_layout = QHBoxLayout(header_frame)
        
        # Create a label with text formatted as a logo
        logo_label = QLabel("RAG Security")
        logo_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #0078d7;
        """)
        
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setContentsMargins(0, 0, 0, 0)
        
        self.status_label = QLabel("Status: Not Connected")
        self.status_label.setStyleSheet("color: #fc5a5a; font-weight: bold;")
        
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter Gemini API Key")
        self.api_key_input.setText(self.api_key)
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        connect_btn = QPushButton("Connect")
        connect_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 5px;
                padding: 8px 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1982db;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
        """)
        connect_btn.clicked.connect(self.initialize_model)
        
        api_layout = QHBoxLayout()
        api_layout.addWidget(self.api_key_input)
        api_layout.addWidget(connect_btn)
        
        status_layout.addWidget(self.status_label)
        status_layout.addLayout(api_layout)
        
        header_layout.addWidget(logo_label)
        header_layout.addStretch()
        header_layout.addWidget(status_container)
        
        # Content tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background-color: #252525;
                color: #ffffff;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                padding: 8px 15px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d7;
            }
            QTabWidget::pane {
                border: 1px solid #333333;
                border-radius: 5px;
                background-color: #1f1f1f;
            }
        """)
        
        # Create tabs
        self.create_query_tab()
        self.create_document_tab()
        
        # Status bar at bottom
        self.statusBar().setStyleSheet("color: #cccccc; background-color: #252525;")
        self.statusBar().showMessage("Ready")
        
        # Add widgets to main layout
        main_layout.addWidget(header_frame)
        main_layout.addWidget(self.tabs)
        
        # Initialize system notifications
        self.show_notification("Welcome to RAG Security Center", 
                             "Please connect to the Gemini API to get started.")
        
    def setup_dark_theme(self):
        # Apply dark theme to the entire application
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #191919;
                color: #ffffff;
            }
            QLineEdit, QTextEdit {
                background-color: #252525;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:pressed {
                background-color: #222222;
            }
            QListWidget {
                background-color: #252525;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #333333;
                border-radius: 5px;
                background-color: #252525;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #0078d7;
                border-radius: 4px;
            }
            QSplitter::handle {
                background-color: #333333;
            }
            QLabel {
                color: #ffffff;
            }
        """)
    
    def create_query_tab(self):
        query_tab = QWidget()
        query_layout = QVBoxLayout(query_tab)
        query_layout.setContentsMargins(15, 15, 15, 15)
        query_layout.setSpacing(10)
        
        # Top section
        top_frame = DarkFrame()
        top_layout = QVBoxLayout(top_frame)
        
        # Query input section
        query_label = QLabel("Ask a question:")
        query_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your question here...")
        self.query_input.setStyleSheet("padding: 12px;")
        
        query_button = QPushButton("Get Answer")
        query_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 5px;
                padding: 10px 15px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1982db;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
        """)
        query_button.clicked.connect(self.get_answer)
        self.query_input.returnPressed.connect(query_button.click)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        
        query_controls = QHBoxLayout()
        query_controls.addWidget(self.query_input)
        query_controls.addWidget(query_button)
        
        top_layout.addWidget(query_label)
        top_layout.addLayout(query_controls)
        top_layout.addWidget(self.progress_bar)
        
        # Results area
        results_frame = DarkFrame()
        results_layout = QVBoxLayout(results_frame)
        
        # Create splitter for answer and context
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        
        # Answer section
        answer_widget = QWidget()
        answer_layout = QVBoxLayout(answer_widget)
        answer_layout.setContentsMargins(0, 0, 0, 0)
        
        answer_label = QLabel("Answer:")
        answer_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        self.answer_output.setPlaceholderText("The answer will appear here...")
        self.answer_output.setStyleSheet("font-size: 13px;")
        
        answer_layout.addWidget(answer_label)
        answer_layout.addWidget(self.answer_output)
        
        # Context section
        context_widget = QWidget()
        context_layout = QVBoxLayout(context_widget)
        context_layout.setContentsMargins(0, 0, 0, 0)
        
        context_label = QLabel("Context Sources:")
        context_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.context_list = QListWidget()
        self.context_list.itemClicked.connect(self.show_context_detail)
        
        context_layout.addWidget(context_label)
        context_layout.addWidget(self.context_list)
        
        # Add widgets to splitter
        splitter.addWidget(answer_widget)
        splitter.addWidget(context_widget)
        splitter.setSizes([300, 150])
        
        results_layout.addWidget(splitter)
        
        # Add main sections to layout
        query_layout.addWidget(top_frame)
        query_layout.addWidget(results_frame)
        
        self.tabs.addTab(query_tab, "Security Intelligence")
    
    def create_document_tab(self):
        doc_tab = QWidget()
        doc_layout = QVBoxLayout(doc_tab)
        doc_layout.setContentsMargins(15, 15, 15, 15)
        doc_layout.setSpacing(10)
        
        # Document management section
        doc_mgmt_frame = DarkFrame()
        doc_mgmt_layout = QVBoxLayout(doc_mgmt_frame)
        
        # Add document section
        add_doc_label = QLabel("Add Knowledge Documents:")
        add_doc_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        # Text input area
        self.document_input = QTextEdit()
        self.document_input.setPlaceholderText("Enter document text here...")
        self.document_input.setMinimumHeight(100)
        
        # Buttons for document actions
        doc_buttons_layout = QHBoxLayout()
        
        add_text_btn = QPushButton("Add Text")
        add_text_btn.clicked.connect(self.add_document_text)
        
        add_file_btn = QPushButton("Add File")
        add_file_btn.clicked.connect(self.add_document_file)
        
        add_samples_btn = QPushButton("Add Samples")
        add_samples_btn.clicked.connect(self.add_sample_documents)
        
        doc_buttons_layout.addWidget(add_text_btn)
        doc_buttons_layout.addWidget(add_file_btn)
        doc_buttons_layout.addWidget(add_samples_btn)
        
        doc_mgmt_layout.addWidget(add_doc_label)
        doc_mgmt_layout.addWidget(self.document_input)
        doc_mgmt_layout.addLayout(doc_buttons_layout)
        
        # Document list section
        doc_list_frame = DarkFrame()
        doc_list_layout = QVBoxLayout(doc_list_frame)
        
        doc_list_label = QLabel("Knowledge Base:")
        doc_list_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.document_list = QListWidget()
        self.document_list.itemClicked.connect(self.show_document_detail)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_document_list)
        
        doc_list_layout.addWidget(doc_list_label)
        doc_list_layout.addWidget(self.document_list)
        doc_list_layout.addWidget(refresh_btn)
        
        # Add main sections to layout
        doc_layout.addWidget(doc_mgmt_frame, 1)
        doc_layout.addWidget(doc_list_frame, 2)
        
        self.tabs.addTab(doc_tab, "Knowledge Management")
    
    def initialize_model(self):
        api_key = self.api_key_input.text().strip()
        if not api_key:
            self.show_notification("Error", "API key is required", is_error=True)
            return
        
        try:
            self.rag_model = SimpleRAGModel(api_key=api_key)
            self.status_label.setText("Status: Connected")
            self.status_label.setStyleSheet("color: #4cd964; font-weight: bold;")
            self.refresh_document_list()
            self.show_notification("Connected", "Successfully connected to Gemini API")
        except Exception as e:
            self.show_notification("Connection Error", f"Failed to initialize: {str(e)}", is_error=True)
            self.status_label.setText("Status: Error")
            self.status_label.setStyleSheet("color: #fc5a5a; font-weight: bold;")
    
    def get_answer(self):
        if not self.rag_model:
            self.show_notification("Error", "Please connect to the API first", is_error=True)
            return
        
        question = self.query_input.text().strip()
        if not question:
            return
        
        # Check if we have documents
        if not self.rag_model.documents or not self.rag_model.is_vectorizer_fitted:
            self.show_notification("Warning", "No documents in knowledge base. Add some documents first or use the 'Add Samples' button.", is_error=False)
        
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.statusBar().showMessage("Getting answer...")
        
        # Clear previous results
        self.answer_output.clear()
        self.context_list.clear()
        
        # Create worker thread
        self.worker = WorkerThread(self.rag_model, question)
        self.worker.update_signal.connect(self.update_answer)
        self.worker.error_signal.connect(self.show_error)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.start()
    
    def update_answer(self, answer, contexts):
        self.answer_output.setText(answer)
        self.statusBar().showMessage("Answer generated")
        self.progress_bar.setVisible(False)
        
        # Update context list
        self.context_list.clear()
        for i, (doc, source, score) in enumerate(contexts):
            # Truncate long documents for display
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            item = QListWidgetItem(f"Source {i+1}: {source} (Score: {score:.2f})")
            item.setData(Qt.ItemDataRole.UserRole, doc)  # Store full text
            self.context_list.addItem(item)
    
    def show_error(self, error_message):
        self.statusBar().showMessage("Error")
        self.progress_bar.setVisible(False)
        self.show_notification("Error", error_message, is_error=True)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value >= 100:
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(False)
    
    def add_document_text(self):
        if not self.rag_model:
            self.show_notification("Error", "Please connect to the API first", is_error=True)
            return
        
        text = self.document_input.toPlainText().strip()
        if not text:
            return
        
        try:
            self.rag_model.add_documents([text], ["manual_entry"])
            self.document_input.clear()
            self.refresh_document_list()
            self.show_notification("Document Added", "Text document has been added to the knowledge base")
        except Exception as e:
            self.show_notification("Error", f"Failed to add document: {str(e)}", is_error=True)
    
    def add_document_file(self):
        if not self.rag_model:
            self.show_notification("Error", "Please connect to the API first", is_error=True)
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Document", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                self.rag_model.add_document_from_file(file_path)
                self.refresh_document_list()
                self.show_notification("Document Added", f"File '{Path(file_path).name}' has been added")
            except Exception as e:
                self.show_notification("Error", f"Failed to add file: {str(e)}", is_error=True)
    
    def add_sample_documents(self):
        if not self.rag_model:
            self.show_notification("Error", "Please connect to the API first", is_error=True)
            return
        
        try:
            count = self.rag_model.initialize_with_samples()
            self.refresh_document_list()
            self.show_notification("Samples Added", f"Added {count} sample documents to the knowledge base")
        except Exception as e:
            self.show_notification("Error", f"Failed to add samples: {str(e)}", is_error=True)
    
    def refresh_document_list(self):
        if not self.rag_model:
            return
        
        self.document_list.clear()
        for i, (doc, source) in enumerate(zip(self.rag_model.documents, self.rag_model.doc_sources)):
            # Truncate long documents for display
            preview = doc[:100] + "..." if len(doc) > 100 else doc
            item = QListWidgetItem(f"Document {i+1}: {source}")
            item.setData(Qt.ItemDataRole.UserRole, doc)  # Store full text
            self.document_list.addItem(item)
    
    def show_document_detail(self, item):
        text = item.data(Qt.ItemDataRole.UserRole)
        if text:
            detail_dialog = QMessageBox(self)
            detail_dialog.setWindowTitle("Document Content")
            detail_dialog.setText(text)
            detail_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            detail_dialog.setStyleSheet("""
                QMessageBox {
                    background-color: #252525;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
            """)
            detail_dialog.exec()
    
    def show_context_detail(self, item):
        text = item.data(Qt.ItemDataRole.UserRole)
        if text:
            detail_dialog = QMessageBox(self)
            detail_dialog.setWindowTitle("Context Content")
            detail_dialog.setText(text)
            detail_dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
            detail_dialog.setStyleSheet("""
                QMessageBox {
                    background-color: #252525;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0078d7;
                    color: white;
                    padding: 6px 12px;
                    border-radius: 4px;
                }
            """)
            detail_dialog.exec()
    
    def show_notification(self, title, message, is_error=False):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        
        if is_error:
            msg_box.setIcon(QMessageBox.Icon.Critical)
        else:
            msg_box.setIcon(QMessageBox.Icon.Information)
        
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #252525;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d7;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
            }
        """)
        
        msg_box.exec()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Use Fusion style for better dark theme support
    window = RAGSecurityApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
