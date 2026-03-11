import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os

# Transformers
from transformers import BertTokenizer, TFBertForSequenceClassification
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


class MedicalModel:

    def __init__(self, max_words=5000, max_len=150):

        self.max_words = max_words
        self.max_len = max_len

        self.tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        self.label_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(max_features=max_words)

        self.baseline_model = LogisticRegression(max_iter=1000)

        self.lstm_model = None
        self.bert_tokenizer = None
        self.bert_model = None

        self.best_model_type = "LSTM"


    def train_baseline(self, X, y_encoded):

        X_tfidf = self.tfidf.fit_transform(X)

        self.baseline_model.fit(X_tfidf, y_encoded)

        y_pred = self.baseline_model.predict(X_tfidf)

        return accuracy_score(y_encoded, y_pred)


    def build_lstm(self, num_classes):

        model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            SpatialDropout1D(0.3),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.lstm_model = model

        return model


    def build_bert(self, num_classes):

        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.bert_model = TFBertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_classes,
            use_safetensors=False
        )

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.bert_model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy']
        )

        return True


    def train_and_compare(self, X, y):

        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded_orig = y_encoded.copy()

        num_classes = len(self.label_encoder.classes_)

        baseline_acc = self.train_baseline(X, y_encoded_orig)

        self.tokenizer.fit_on_texts(X)

        sequences = self.tokenizer.texts_to_sequences(X)

        X_padded = pad_sequences(sequences, maxlen=self.max_len)

        if len(X_padded) < 20:

            X_padded_lstm = np.tile(X_padded, (5, 1))
            y_encoded_lstm = np.tile(y_encoded_orig, (5,))

        else:

            X_padded_lstm = X_padded
            y_encoded_lstm = y_encoded_orig

        X_train, X_val, y_train, y_val = train_test_split(
            X_padded_lstm,
            y_encoded_lstm,
            test_size=0.2,
            random_state=42
        )

        self.build_lstm(num_classes)

        history = self.lstm_model.fit(
            X_train,
            y_train,
            epochs=5,
            validation_data=(X_val, y_val),
            verbose=0
        )

        lstm_acc = max(history.history['val_accuracy'])

        self.build_bert(num_classes)

        bert_inputs = self.bert_tokenizer(
            list(X),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='tf'
        )

        bert_history = self.bert_model.fit(
            dict(bert_inputs),
            y_encoded_orig,
            epochs=3,
            batch_size=4,
            verbose=0
        )

        bert_acc = bert_history.history['accuracy'][-1]

        accuracies = {
            "Baseline": baseline_acc,
            "LSTM": lstm_acc,
            "BERT": bert_acc
        }

        self.best_model_type = max(accuracies, key=accuracies.get)

        return {
            "baseline_accuracy": float(baseline_acc),
            "lstm_accuracy": float(lstm_acc),
            "bert_accuracy": float(bert_acc),
            "best_model": self.best_model_type,
            "f1_score": float(
                f1_score(
                    y_encoded_orig,
                    self.baseline_model.predict(self.tfidf.transform(X)),
                    average='weighted'
                )
            )
        }


    def predict(self, text, user_metadata=None):
        combined_text = text
        if user_metadata:
            combined_text += f" {user_metadata.get('age', '')} {user_metadata.get('gender', '')}"
            
        if self.best_model_type == "BERT" and self.bert_model is not None:
            inputs = self.bert_tokenizer([combined_text], truncation=True, padding=True, max_length=128, return_tensors='tf')
            logits = self.bert_model(inputs).logits
            probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
            
        elif self.best_model_type == "Baseline":
            X_tfidf = self.tfidf.transform([combined_text])
            probs = self.baseline_model.predict_proba(X_tfidf)[0]
            
        else:
            seq = self.tokenizer.texts_to_sequences([combined_text])
            padded = pad_sequences(seq, maxlen=self.max_len)
            probs = self.lstm_model.predict(padded, verbose=0)[0]
        
        top_idx = np.argsort(probs)[-3:][::-1]
        results = []
        for idx in top_idx:
            condition = str(self.label_encoder.inverse_transform([idx])[0])
            
            # حساب النسبة المئوية بشكل صحيح
            prob_value = float(probs[idx])
            
            results.append({
                "condition": condition,
                "probability": round(prob_value, 4) # نرسل القيمة كـ decimal والواجهة تضرب في 100
            })
            
        return results # اتأكد إن السطر ده في نفس مستوى الـ for مش جواها

    def save(self, path_prefix):

        if self.lstm_model:

            self.lstm_model.save(f"{path_prefix}_model.h5")

        with open(f"{path_prefix}_tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)

        with open(f"{path_prefix}_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(f"{path_prefix}_tfidf.pkl", "wb") as f:
            pickle.dump(self.tfidf, f)

        with open(f"{path_prefix}_baseline.pkl", "wb") as f:
            pickle.dump(self.baseline_model, f)

        if self.bert_model and self.best_model_type == "BERT":

            self.bert_model.save_pretrained(
                f"{path_prefix}_bert_model",
                safe_serialization=False
            )

            self.bert_tokenizer.save_pretrained(
                f"{path_prefix}_bert_model"
            )


    def load(self, path_prefix):

        if os.path.exists(f"{path_prefix}_model.h5"):

            self.lstm_model = tf.keras.models.load_model(
                f"{path_prefix}_model.h5"
            )

        with open(f"{path_prefix}_tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)

        with open(f"{path_prefix}_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)

        if os.path.exists(f"{path_prefix}_tfidf.pkl"):

            with open(f"{path_prefix}_tfidf.pkl", "rb") as f:
                self.tfidf = pickle.load(f)

            with open(f"{path_prefix}_baseline.pkl", "rb") as f:
                self.baseline_model = pickle.load(f)

        if os.path.exists(f"{path_prefix}_bert_model"):

            self.bert_model = TFBertForSequenceClassification.from_pretrained(
                f"{path_prefix}_bert_model",
                use_safetensors=False
            )

            self.bert_tokenizer = BertTokenizer.from_pretrained(
                f"{path_prefix}_bert_model"
            )