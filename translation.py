# Install necessary libraries
#!pip install transformers
#!pip install -U transformers
#!pip install sentencepiece

import os
import urllib.request
import zipfile
import tensorflow as tf
from transformers import MarianTokenizer, MarianMTModel

class TranslationModel:
    def __init__(self, model_name, dataset_url, max_length=128, batch_size=32, num_epochs=5, learning_rate=1e-4):
        self.model_name = model_name
        self.dataset_url = dataset_url
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        
    def download_and_extract_data(self):
        dataset_dir = "spa-eng"
        file_name = "spa-eng.zip"
        
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        urllib.request.urlretrieve(self.dataset_url, os.path.join(dataset_dir, file_name))
        with zipfile.ZipFile(os.path.join(dataset_dir, file_name), 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    
    def load_data(self, file_path):
        lines = open(file_path, encoding='utf-8').read().strip().split('\n')
        pairs = [(line.split('\t')[0], line.split('\t')[1]) for line in lines]
        return pairs
    
    def preprocess_data(self, source_texts, target_texts):
        input_encodings = self.tokenizer(source_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='tf')
        target_encodings = self.tokenizer(target_texts, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='tf')

        inputs = {key: tf.convert_to_tensor(input_encodings[key]) for key in input_encodings}
        targets = tf.convert_to_tensor(target_encodings['input_ids'])
        
        return inputs, targets
    
    def build_train_dataset(self, inputs, targets):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.shuffle(len(inputs)).batch(self.batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
    
    

    def train_step(self, inputs, targets):
        with tf.GradientTape() as tape:
            logits = self.model(**inputs, decoder_input_ids=targets).logits  # Pass inputs as **kwargs
            loss = loss_fn(targets, logits)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def train(self, train_dataset):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            total_loss = 0
            total_batches = 0
            for batch, (inputs, targets) in enumerate(train_dataset):
                loss = self.train_step(inputs, targets)  # Use the regular train step
                total_loss += loss.numpy().item()
                total_batches = batch + 1
                if batch % 100 == 0:
                    print(f"Batch {batch}, Loss: {loss.numpy():.4f}")
            print(f"Epoch Loss: {total_loss / total_batches:.4f}")


if __name__ == "__main__":
    # Instantiate the TranslationModel and start the training process
    model = TranslationModel(
        model_name="Helsinki-NLP/opus-mt-es-en",
        dataset_url="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
        max_length=128,
        batch_size=32,
        num_epochs=5,
        learning_rate=1e-4
    )

    model.download_and_extract_data()
    train_data = model.load_data(os.path.join("spa-eng", "spa-eng", "spa.txt"))
    source_texts, target_texts = zip(*train_data)
    inputs, targets = model.preprocess_data(source_texts, target_texts)
    train_dataset = model.build_train_dataset(inputs, targets)
    model.train(train_dataset)
