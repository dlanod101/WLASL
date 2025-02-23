import json
import tensorflow as tf # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, Concatenate # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np

def load_dataset(json_path):
    """Load and preprocess the dataset."""
    print("Loading dataset...")
    with open(json_path, "r") as f:
        return json.load(f)

def prepare_data(dataset, num_words=10000, max_length=20):
    """Prepare and tokenize the data."""
    print("Preparing data...")
    
    # Extract text pairs
    english_texts = [entry["english"].lower() for entry in dataset]
    gloss_texts = [entry["gloss"].upper() for entry in dataset]

    # Initialize tokenizers
    tokenizer_eng = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer_gloss = Tokenizer(num_words=num_words, oov_token="<OOV>")

    # Fit tokenizers
    print("Fitting tokenizers...")
    tokenizer_eng.fit_on_texts(english_texts)
    tokenizer_gloss.fit_on_texts(gloss_texts)

    # Convert to sequences
    eng_sequences = tokenizer_eng.texts_to_sequences(english_texts)
    gloss_sequences = tokenizer_gloss.texts_to_sequences(gloss_texts)

    # Pad sequences
    eng_padded = pad_sequences(eng_sequences, maxlen=max_length, padding="post")
    gloss_padded = pad_sequences(gloss_sequences, maxlen=max_length, padding="post")

    print(f"Vocabulary sizes - English: {len(tokenizer_eng.word_index)}, Gloss: {len(tokenizer_gloss.word_index)}")
    return eng_padded, gloss_padded, tokenizer_eng, tokenizer_gloss

def build_seq2seq_model(num_words, embedding_dim=256, hidden_units=512, max_length=20, dropout_rate=0.3):
    """Build an improved sequence-to-sequence model."""
    print("Building model...")
    
    # Encoder
    encoder_inputs = Input(shape=(max_length,))
    encoder_embedding = Embedding(num_words, embedding_dim)(encoder_inputs)
    encoder_embedding = Dropout(dropout_rate)(encoder_embedding)
    
    # Use bidirectional LSTM for encoder
    forward_lstm = LSTM(hidden_units, return_state=True)
    backward_lstm = LSTM(hidden_units, return_state=True, go_backwards=True)
    
    # Forward LSTM
    forward_outputs, forward_h, forward_c = forward_lstm(encoder_embedding)
    
    # Backward LSTM
    backward_outputs, backward_h, backward_c = backward_lstm(encoder_embedding)
    
    # Concatenate states using Keras layer
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(max_length,))
    decoder_embedding = Embedding(num_words, embedding_dim)(decoder_inputs)
    decoder_embedding = Dropout(dropout_rate)(decoder_embedding)
    
    # Regular LSTM for decoder
    decoder_lstm = LSTM(hidden_units * 2, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_outputs = Dropout(dropout_rate)(decoder_outputs)
    
    # Dense output layer
    decoder_dense = Dense(num_words, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Create and compile model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    # Configuration
    dataset_path = "text_to_gloss.json"
    max_length = 20
    num_words = 10000
    epochs = 1
    batch_size = 64
    validation_split = 0.2

    # Load and prepare data
    dataset = load_dataset(dataset_path)
    eng_padded, gloss_padded, tokenizer_eng, tokenizer_gloss = prepare_data(
        dataset, num_words, max_length
    )

    # Build model
    model = build_seq2seq_model(num_words, max_length=max_length)
    print(model.summary())

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]

    # Train model
    print("\nStarting training...")
    history = model.fit(
        [eng_padded, gloss_padded],
        gloss_padded,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks
    )

    # Save model and tokenizers
    print("\nSaving model and tokenizers...")
    model.save("text_to_gloss_tf_model.keras")
    
    with open("tokenizer_eng.json", "w") as f:
        json.dump(tokenizer_eng.word_index, f)
    with open("tokenizer_gloss.json", "w") as f:
        json.dump(tokenizer_gloss.word_index, f)

    print("Training complete. Model and tokenizers saved.")
