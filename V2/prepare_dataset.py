import json
import random

def load_wlasl_data(json_path, class_list_path):
    """Load WLASL dataset and class list."""
    print("Loading WLASL data...")
    
    # Load class list
    valid_words = set()
    with open(class_list_path, 'r') as f:
        for line in f:
            # Remove number prefix, tabs, and whitespace
            word = line.strip().lower()
            word = ' '.join(word.split()[1:]) if word.split()[0].rstrip('.').isdigit() else word
            word = word.replace('\t', '')
            valid_words.add(word)
    
    # Load WLASL data
    with open(json_path, "r") as f:
        data = json.load(f)
    
    return data, valid_words

def create_training_pairs(wlasl_data, valid_words):
    """Create training pairs from actual WLASL data."""
    dataset = []
    
    # Create single word pairs
    for entry in wlasl_data:
        gloss = entry["gloss"]
        # Convert gloss to lowercase for matching
        english_word = gloss.lower()
        
        if english_word in valid_words:
            # Add direct word-to-gloss mapping
            dataset.append({
                "english": english_word,
                "gloss": gloss.upper()
            })
            
            # Add with "the" prefix
            dataset.append({
                "english": f"the {english_word}",
                "gloss": gloss.upper()
            })
            
            # Add with "a" prefix for appropriate words
            if english_word[0] not in 'aeiou':
                dataset.append({
                    "english": f"a {english_word}",
                    "gloss": gloss.upper()
                })
    
    # Create some simple phrase combinations
    glosses = [entry["gloss"] for entry in wlasl_data]
    for i in range(len(dataset)):
        if random.random() < 0.3:  # 30% chance to create a combination
            if i + 1 < len(dataset):
                # Combine two consecutive entries
                dataset.append({
                    "english": f"{dataset[i]['english']} {dataset[i+1]['english']}",
                    "gloss": f"{dataset[i]['gloss']} {dataset[i+1]['gloss']}"
                })
    
    return dataset

def save_dataset(dataset, output_path):
    """Save the dataset with examples."""
    random.shuffle(dataset)
    
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    
    print("\nExample entries from the dataset:")
    for i in range(min(5, len(dataset))):
        print(f"\nEnglish: {dataset[i]['english']}")
        print(f"Gloss: {dataset[i]['gloss']}")

if __name__ == "__main__":
    json_path = r"wlasl_data\WLASL_v0.3.json"
    class_list_path = r"wlasl_data\wlasl_class_list.txt"
    output_path = "text_to_gloss.json"
    
    # Load data
    wlasl_data, valid_words = load_wlasl_data(json_path, class_list_path)
    print(f"Loaded {len(wlasl_data)} entries and {len(valid_words)} valid words")
    
    # Create dataset
    print("\nCreating training pairs...")
    dataset = create_training_pairs(wlasl_data, valid_words)
    print(f"Generated {len(dataset)} pairs")
    
    # Save dataset
    print("\nSaving dataset...")
    save_dataset(dataset, output_path)
    print(f"Dataset saved as {output_path}")
