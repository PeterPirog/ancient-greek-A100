import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from pathlib import Path
from transformers import TFTrainer, TFTrainingArguments

# Ścieżka do katalogu z danymi treningowymi
data_directory = "/home/ppirog/projects/ancient-greek-A100/corpus_mini"

# Pobierz listę plików tekstowych w katalogu i jego podkatalogach
paths = [str(x) for x in Path(data_directory).glob("**/*.txt")]

# Inicjalizacja modelu RoBERTa i jego tokenizera
model_name = 'bert-base-cased'  # Możesz zmienić na odpowiednią wersję modelu RoBERTa
model = TFAutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Przygotowanie danych treningowych
input_texts = []

# Wczytaj zawartość plików tekstowych do danych treningowych
for path in paths:
    with open(path, "r", encoding="utf-8") as file:
        input_texts.append(file.read())

# Tokenizacja i kodowanie danych treningowych
max_seq_length = 512

# Funkcja do przycinania lub uzupełniania sekwencji do maksymalnej długości
def pad_or_truncate_sequence(input_ids):
    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]  # Przyciąć do maksymalnej długości
    return input_ids

# Tokenizacja i kodowanie danych treningowych
inputs = tokenizer(input_texts, padding='max_length', truncation=True, max_length=max_seq_length, return_tensors="tf")

dataset = tf.data.Dataset.from_tensor_slices(inputs)
dataset = dataset.map(lambda x: {'input_ids': pad_or_truncate_sequence(x['input_ids']), 'attention_mask': x['attention_mask']})
dataset = dataset.shuffle(100).padded_batch(4, padded_shapes={'input_ids': [max_seq_length], 'attention_mask': [max_seq_length]}, drop_remainder=True)

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
