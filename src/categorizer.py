import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class NewsCategorizer:
    def __init__(self, training_data_dir):
        """
        Initialize categorizer with training data from text files
        
        Args:
            training_data_dir (str): Directory containing training text files
        """
        self.training_data = self._load_training_data(training_data_dir)
        self.stop_words = set(stopwords.words('english'))
    
    def _load_training_data(self, training_dir):
        """
        Load training data from text files in the specified directory
        
        Args:
            training_dir (str): Directory containing category text files
        
        Returns:
            dict: Training data for each category
        """
        training_data = {}
        
        # Ensure training directory exists
        if not os.path.exists(training_dir):
            raise ValueError(f"Training data directory not found: {training_dir}")
        
        # Read training files for each category
        for filename in os.listdir(training_dir):
            if filename.endswith('.txt'):
                category = os.path.splitext(filename)[0].capitalize()
                file_path = os.path.join(training_dir, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        # Split file content into individual training examples
                        training_examples = file.read().split('\n')
                        # Remove empty lines
                        training_examples = [line.strip() for line in training_examples if line.strip()]
                        training_data[category] = training_examples
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        return training_data
    
    def preprocess_text(self, text):
        """
        Preprocess the input text
        
        Args:
            text (str): Input text to preprocess
        
        Returns:
            str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def train_model(self):
        """
        Train a text classification model
        
        Returns:
            Pipeline: Trained machine learning model
        """
        # Prepare training data
        X_train = []
        y_train = []
        for category, texts in self.training_data.items():
            X_train.extend([self.preprocess_text(text) for text in texts])
            y_train.extend([category] * len(texts))
        
        # Create and train the pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
        model.fit(X_train, y_train)
        
        return model

def categorize_files(directory, training_data_dir, model):
    """
    Categorize text files in the given directory
    
    Args:
        directory (str): Directory containing files to categorize
        training_data_dir (str): Directory containing training data
        model (Pipeline): Trained classification model
    """
    # Create categorizer
    categorizer = NewsCategorizer(training_data_dir)
    
    # Validate directory
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return
    
    # Get all text files
    text_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    
    if not text_files:
        print("No text files found in the directory.")
        return
    
    # Categorize each file
    print("\n--- Article Categorization ---")
    for file_name in text_files:
        file_path = os.path.join(directory, file_name)
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            
            # Preprocess and categorize
            preprocessed_text = categorizer.preprocess_text(text)
            category = model.predict([preprocessed_text])[0]
            
            # Print results
            print(f"File: {file_name} | Category: {category}")
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

def main():
    # Set up directory paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_data_dir = os.path.join(base_dir, '..', 'data', 'training')
    input_directory = os.path.join(base_dir, '..', 'data', 'test', 'sample_articles')
    
    # Ensure input directory exists
    os.makedirs(input_directory, exist_ok=True)
    
    # Create and train the model
    categorizer = NewsCategorizer(training_data_dir)
    model = categorizer.train_model()
    
    # Categorize files
    categorize_files(input_directory, training_data_dir, model)

if __name__ == "__main__":
    main()