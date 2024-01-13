import tensorflow as tf
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
from pathlib import Path
from transformers import TFTrainer, TFTrainingArguments  # Dodajemy importy dla TFTrainer i TFTrainingArguments

# Ścieżka do katalogu z danymi treningowymi
data_directory = "./corpus"

# Pobierz listę plików tekstowych w katalogu i jego podkatalogach
paths = [str(x) for x in Path(data_directory).glob("**/*.txt")]

# Inicjalizacja modelu RoBERTa i jego tokenizera
model_name = 'bert-base-cased'  # Możesz zmienić na odpowiednią wersję modelu RoBERTa
model = TFAutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Przygotowanie danych treningowych
input_texts = []

# Wczytaj zawartość plików tekstowych do danych treningowych
for path in paths:
    with open(path, "r", encoding="utf-8") as file:
        input_texts.append(file.read())

# Tokenizacja i kodowanie danych treningowych
inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="tf")

# Utworzenie datasetu TensorFlow
dataset = tf.data.Dataset.from_tensor_slices(inputs)
dataset = dataset.shuffle(100).batch(4)

# Parametry treningowe
learning_rate = 1e-5
num_epochs = 3

# Konfiguracja trenera
training_args = TFTrainingArguments(
    output_dir="./roberta-training",
    per_device_train_batch_size=4,
    save_steps=10_000,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Trening modelu
trainer.train()

# Zapisz wytrenowany model
model.save_pretrained("./roberta-model")
