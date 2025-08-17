import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

class TextClassifier:
    def __init__(self, model, tokenizer, label_encoder):
        """
        مصنف النصوص باستخدام نموذج LSTM
        
        Args:
            model: النموذج المدرب
            tokenizer: محول النصوص
            label_encoder: محول التصنيفات
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_len = 100  # نفس القيمة المستخدمة في التدريب
        
        # قائمة الفئات المتاحة
        self.categories = {
            1: "رياضة",
            2: "سياسة", 
            3: "تقنية",
            4: "اقتصاد",
            5: "ثقافة",
            6: "صحة",
            7: "تعليم",
            8: "أخرى"
        }
    
    def preprocess_text(self, text):
        """
        معالجة النص قبل التصنيف
        
        Args:
            text (str): النص المراد معالجته
            
        Returns:
            list: قائمة نتائج التصنيف
        """
        results = []
        
        # معالجة النصوص
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # تحويل إلى تسلسلات
        sequences = self.tokenizer.texts_to_sequences(processed_texts)
        
        # إضافة padding
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_len, 
            padding='post', 
            truncating='post'
        )
        
        # التنبؤ
        predictions = self.model.predict(padded_sequences, verbose=0)
        
        # معالجة النتائج
        for i, pred in enumerate(predictions):
            predicted_class_idx = np.argmax(pred)
            confidence = pred[predicted_class_idx]
            
            predicted_class_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            predicted_class_arabic = self.categories.get(predicted_class_name, "غير محدد")
            
            results.append({
                'text': texts[i],
                'predicted_class': predicted_class_arabic,
                'predicted_class_id': int(predicted_class_name),
                'confidence': float(confidence)
            })
        
        return results
    
    def get_model_info(self):
        """
        معلومات عن النموذج
        
        Returns:
            dict: معلومات النموذج
        """
        return {
            'model_type': 'LSTM',
            'vocab_size': len(self.tokenizer.word_index),
            'max_sequence_length': self.max_len,
            'num_classes': len(self.categories),
            'categories': self.categories
        }
            str: النص بعد المعالجة
        """
        if not isinstance(text, str):
            text = str(text)
        
        # إزالة العلامات والأرقام الزائدة
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\w]', ' ', text)
        
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, text):
        """
        تصنيف النص
        
        Args:
            text (str): النص المراد تصنيفه
            
        Returns:
            dict: نتائج التصنيف
        """
        # معالجة النص
        processed_text = self.preprocess_text(text)
        
        # تحويل النص إلى تسلسل
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        
        # إضافة padding
        padded_sequence = pad_sequences(
            sequence, 
            maxlen=self.max_len, 
            padding='post', 
            truncating='post'
        )
        
        # التنبؤ
        predictions = self.model.predict(padded_sequence, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        # تحويل الفهرس إلى اسم الفئة
        predicted_class_name = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # الحصول على اسم الفئة بالعربية
        predicted_class_arabic = self.categories.get(predicted_class_name, "غير محدد")
        
        # جميع الاحتماليات
        all_probabilities = predictions[0].tolist()
        all_classes = []
        
        for i, prob in enumerate(all_probabilities):
            class_idx = self.label_encoder.inverse_transform([i])[0]
            class_name = self.categories.get(class_idx, f"فئة {class_idx}")
            all_classes.append(class_name)
        
        return {
            'predicted_class': predicted_class_arabic,
            'predicted_class_id': int(predicted_class_name),
            'confidence': float(confidence),
            'all_probabilities': all_probabilities,
            'all_classes': all_classes,
            'processed_text': processed_text
        }
    
    def predict_batch(self, texts):
        """
        تصنيف مجموعة من النصوص
        
        Args:
            texts (list): قائمة النصوص
            
        Returns: