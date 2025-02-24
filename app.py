import nltk
import os

# Create a data directory if it doesn't exist
#if not os.path.exists(os.path.expanduser('~/nltk_data')):
#    os.makedirs(os.path.expanduser('~/nltk_data'))

# Download required NLTK data
#try:
#    nltk.data.find('corpora/wordnet')
#except LookupError:
#    nltk.download('wordnet')
#try:
#    nltk.data.find('taggers/averaged_perceptron_tagger')
#except LookupError:
    #nltk.download('averaged_perceptron_tagger')

from flask import Flask, request, jsonify, send_file, render_template
from predict import WLASLRetrieval
from nltk.corpus import wordnet
import speech_recognition as sr

app = Flask(__name__)

# Initialize the WLASL retriever
json_path = r"wlasl_data/WLASL_v0.3.json"
video_dir = r"wlasl_data/new_refined_videos"
class_list_path = r"wlasl_data/wlasl_class_list.txt"

# Initialize the speech recognizer
recognizer = sr.Recognizer()

print("Initializing WLASL retriever...")
retriever = WLASLRetrieval(json_path, video_dir, class_list_path)
print("Retriever initialized successfully")

def get_synonyms(word):
    """Get all synonyms for a word"""
    # Special cases
    word = word.lower()
    if word == 'am':
        return ['represent']
    elif word == 'hi':
        return ['hello']
    elif word == 'is':
        return ['']
    elif word == 'the':
        return ['']
    elif word == 'and':
        return ['']
        
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().lower()
            if synonym != 'cost':  # Exclude 'cost' from synonyms
                synonyms.add(synonym)
    return list(synonyms)

def find_matching_word(word, available_words):
    """Find a matching word or its synonym in available words"""
    word = word.lower()
    # First try exact match
    if word in available_words:
        return word
    
    # Then try synonyms
    synonyms = get_synonyms(word)
    for synonym in synonyms:
        if synonym in available_words:
            return synonym
    
    return None

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    """
    Endpoint to translate text to sign language videos
    Expects JSON: {"text": "your text here"}
    Returns: {"videos": ["path1", "path2", ...]}
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
            
        # Clean the text: remove symbols and extra spaces
        text = ''.join(c for c in data['text'] if c.isalnum() or c.isspace())
        text = ' '.join(text.split())  # Remove extra spaces
        
        # Get available words
        available_words = set(retriever.word_to_index.keys())
        
        # Process each word and find matches including synonyms
        words = text.lower().split()
        matched_words = []
        original_to_matched = {}  # Map original words to their matches
        
        for word in words:
            matched_word = find_matching_word(word, available_words)
            if matched_word:
                matched_words.append(matched_word)
                original_to_matched[word] = matched_word
        
        # Get videos for matched words
        video_paths = []
        for word in matched_words:
            paths = retriever.get_video_path(word)
            if paths:
                video_paths.append(paths)
        
        # Convert video paths to URLs or relative paths as needed
        videos = [os.path.basename(path) for path in video_paths if path]
        
        return jsonify({
            'text': ' '.join(matched_words),
            'videos': videos,
            'count': len(videos),
            'mappings': original_to_matched  # Include word mappings for frontend
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/available-words', methods=['GET'])
def get_available_words():
    """
    Endpoint to get list of available words
    Returns: {"words": ["word1", "word2", ...]}
    """
    try:
        words = list(retriever.word_to_index.keys())
        return jsonify({
            'words': words,
            'count': len(words)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/videos/<video_id>', methods=['GET'])
def serve_video(video_id):
    """
    Endpoint to serve video files
    """
    try:
        video_path = os.path.join(video_dir, video_id)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404
            
        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """
    Endpoint to convert speech to text
    Expects: Audio data in the request
    Returns: {"text": "transcribed text"}
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        
        # Convert the audio to text
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            
            return jsonify({
                'success': True,
                'text': text
            })
            
    except sr.UnknownValueError:
        return jsonify({
            'success': False,
            'error': "Couldn't understand the audio"
        }), 400
    except sr.RequestError:
        return jsonify({
            'success': False,
            'error': "Error with the speech recognition service"
        }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 
