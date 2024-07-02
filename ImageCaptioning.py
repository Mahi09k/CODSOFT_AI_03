import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Add, RepeatVector
from tensorflow.keras.utils import to_categorical

# Function to extract features from an image using TensorFlow Hub
def extract_features(image_path, model):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    feature = model(image)
    feature = tf.reshape(feature, (feature.shape[0], -1))  # Flatten the feature vector
    return feature.numpy()

# Load ResNet50V2 model from TensorFlow Hub
def load_resnet_v2_50():
    model = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5", trainable=False)
    return model

# Define the RNN for the decoder
def build_captioning_model(resnet_model, vocab_size, max_length):
    # Dummy input to determine output_shape
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    dummy_feature = resnet_model(dummy_input)
    inputs1 = Input(shape=(int(np.prod(dummy_feature.shape[1:])),))  # Adjust input shape based on ResNet output
    fe1 = Dense(256, activation='relu')(inputs1)
    fe2 = RepeatVector(max_length)(fe1)

    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = LSTM(256, return_sequences=True)(se1)

    decoder1 = Add()([fe2, se2])
    decoder2 = LSTM(256)(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Function to generate captions
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for step in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)

        print(f"Step {step + 1}: Predicted Index: {yhat_index}, Predicted Word: {word}")

        if word is None or word == 'endseq':
            break

        in_text += ' ' + word

    return in_text


# Example usage
def main():
    image_path = 'download.jpeg'  # Replace with your image path
    captions = ["startseq a cat is sitting on the table endseq", "startseq a dog is playing with a ball endseq"]

    # Initialize and fit the tokenizer with your dataset captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size

    # Load ResNet50V2 model
    resnet_model = load_resnet_v2_50()

    # Build captioning model
    max_length = 30  # Adjust as needed
    model = build_captioning_model(resnet_model, vocab_size, max_length)
    model.summary()

    # Extract features from the image
    photo = extract_features(image_path, resnet_model)

    # Generate caption
    caption = generate_caption(model, tokenizer, photo, max_length)
    print("Generated Caption:", caption)

if __name__ == "__main__":
    main()
