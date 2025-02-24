import json
import os

class WLASLRetrieval:
    def __init__(self, json_path, video_dir, class_list_path):
        """
        Initializes the retrieval system.
        """
        self.video_dir = video_dir
        
        # Load the class list
        self.word_to_index = {}
        print("Loading class list...")  # Debug print
        with open(class_list_path, 'r') as f:
            for idx, line in enumerate(f):
                # Remove number prefix, tabs, and whitespace
                word = line.strip().lower()
                # Remove any leading numbers and dots (e.g., "1.", "22.", etc.)
                word = ' '.join(word.split()[1:]) if word.split()[0].rstrip('.').isdigit() else word
                word = word.replace('\t', '')
                self.word_to_index[word] = idx
        print(f"Loaded {len(self.word_to_index)} words")  # Debug print
        
        # Load the JSON data
        print("Loading JSON data...")  # Debug print
        with open(json_path, "r") as f:
            self.raw_data = json.load(f)
        print(f"Loaded {len(self.raw_data)} entries")  # Debug print

    def text_to_glosses(self, text):
        """
        Converts text input into WLASL glosses using the class list.
        """
        words = text.lower().split()
        glosses = []
        print(f"\nLooking for words: {words}")  # Debug print
        for word in words:
            print(f"Checking word: '{word}'")  # Debug print
            if word in self.word_to_index:
                index = self.word_to_index[word]
                print(f"Found index {index} for word '{word}'")  # Debug print
                if index < len(self.raw_data):
                    gloss = self.raw_data[index]['gloss']
                    print(f"Found gloss: {gloss}")  # Debug print
                    glosses.append(gloss)
            else:
                print(f"Word '{word}' not found in class list")  # Debug print
                # Print some available words for reference
                print("Available words (first 10):", list(self.word_to_index.keys())[:10])
        return glosses

    def get_video_path(self, gloss):
        """
        Gets the video path for a given gloss.
        """
        # Special case for 'represent'
        if gloss.lower() == 'represent':
            return os.path.join(self.video_dir, '47387.mp4')

        # Find the entry for this gloss
        for entry in self.raw_data:
            if entry['gloss'] == gloss:  # Exact match since we're using the class list
                # Try each instance until we find a video that exists
                for instance in entry['instances']:
                    video_id = str(instance['video_id'])
                    video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
                    if os.path.exists(video_path):
                        return video_path
        return None

    def retrieve_videos(self, text):
        """
        Retrieves a sequence of videos based on input text.
        """
        glosses = self.text_to_glosses(text)
        print(f"\nFound glosses: {glosses}")  # Debug print
        video_paths = []

        for gloss in glosses:
            video_path = self.get_video_path(gloss)
            if video_path and os.path.exists(video_path):
                video_paths.append(video_path)
            else:
                print(f"Warning: No video found for '{gloss}'")

        return video_paths

if __name__ == "__main__":
    json_path = r"wlasl_data/WLASL_v0.3.json"
    video_dir = r"wlasl_data/refined_videos"
    class_list_path = r"wlasl_data/wlasl_class_list.txt"

    print("\nInitializing retriever...")  # Debug print
    retriever = WLASLRetrieval(json_path, video_dir, class_list_path)

    while True:
        user_input = input("\nEnter text (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
            
        video_sequence = retriever.retrieve_videos(user_input)
        if video_sequence:
            print("Found videos:", video_sequence)
        else:
            print("No matching videos found.")
