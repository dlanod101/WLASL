import json
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import os

class SignPredictor:
    def __init__(self, model_path, tokenizer_eng_path, tokenizer_gloss_path, wlasl_json_path, video_dir, class_list_path):
        print("Initializing SignPredictor...")
        
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
        
        # Load tokenizers
        with open(tokenizer_eng_path, "r") as f:
            tokenizer_eng_index = json.load(f)
        with open(tokenizer_gloss_path, "r") as f:
            tokenizer_gloss_index = json.load(f)
            
        # Initialize tokenizers
        self.tokenizer_eng = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer_eng.word_index = tokenizer_eng_index
        
        self.tokenizer_gloss = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer_gloss.word_index = tokenizer_gloss_index
        self.gloss_index_to_word = {v: k for k, v in tokenizer_gloss_index.items()}
        
        # Add class list loading
        self.word_to_index = {}
        print("Loading class list...")
        with open(class_list_path, 'r') as f:
            for idx, line in enumerate(f):
                word = line.strip().lower()
                word = ' '.join(word.split()[1:]) if word.split()[0].rstrip('.').isdigit() else word
                word = word.replace('\t', '')
                self.word_to_index[word] = idx
        print(f"Loaded {len(self.word_to_index)} words")

        # Load WLASL data
        with open(wlasl_json_path, "r") as f:
            self.wlasl_data = json.load(f)
        
        self.video_dir = video_dir
        self.max_length = 20  # Same as training
        print("Initialization complete")

    def predict_gloss(self, sentence):
        """Convert English text to WLASL gloss sequence using the trained model."""
        # Preprocess input
        input_seq = self.tokenizer_eng.texts_to_sequences([sentence.lower()])
        input_padded = pad_sequences(input_seq, maxlen=self.max_length, padding="post")
        
        # Make prediction
        predictions = self.model.predict(
            [input_padded, np.zeros_like(input_padded)],
            verbose=0
        )
        
        # Convert predictions to gloss sequence
        gloss_sequence = []
        for pred in predictions[0]:
            gloss_index = np.argmax(pred)
            if gloss_index in self.gloss_index_to_word:
                gloss = self.gloss_index_to_word[gloss_index]
                if gloss not in ["<OOV>", "<PAD>"]:  # Skip special tokens
                    gloss_sequence.append(gloss)
        
        return gloss_sequence if gloss_sequence else ["No valid gloss prediction"]

    def get_video_path(self, gloss):
        """Get video path for a gloss using improved matching."""
        # Special case for 'represent'
        if gloss.upper() == "REPRESENT":
            return os.path.join(self.video_dir, "47387.mp4")
        
        # Find the entry for this gloss
        for entry in self.wlasl_data:
            if entry['gloss'].upper() == gloss.upper():
                # Try each instance until we find a video that exists
                for instance in entry['instances']:
                    video_id = str(instance['video_id'])
                    video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                    if os.path.exists(video_path):
                        return video_path
        return None

    def retrieve_videos(self, sentence):
        """Get video paths for a sentence."""
        gloss_sequence = self.predict_gloss(sentence)
        print(f"Predicted Glosses: {gloss_sequence}")  # Debug output
        
        video_paths = []
        for gloss in gloss_sequence:
            video_path = self.get_video_path(gloss)
            if video_path:
                video_paths.append(video_path)
                print(f"Found video for gloss '{gloss}': {video_path}")  # Debug output
            else:
                print(f"No video found for gloss '{gloss}'")  # Debug output
                
        return video_paths

if __name__ == "__main__":
    # Initialize paths
    model_path = "text_to_gloss_tf_model.keras"
    tokenizer_eng_path = "tokenizer_eng.json"
    tokenizer_gloss_path = "tokenizer_gloss.json"
    wlasl_json_path = r"wlasl_data\WLASL_v0.3.json"
    video_dir = r"wlasl_data\videos"
    class_list_path = r"wlasl_data\wlasl_class_list.txt"
    
    # Create predictor with new class_list_path parameter
    predictor = SignPredictor(
        model_path,
        tokenizer_eng_path,
        tokenizer_gloss_path,
        wlasl_json_path,
        video_dir,
        class_list_path
    )
    
    # Interactive testing
    while True:
        user_input = input("\nEnter a sentence (or 'q' to quit): ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
            
        video_paths = predictor.retrieve_videos(user_input)
        if video_paths:
            print("\nFound videos:", video_paths)
        else:
            print("\nNo matching videos found.")
