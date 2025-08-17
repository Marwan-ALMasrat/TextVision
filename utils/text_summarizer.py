import re
import numpy as np
from collections import Counter
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

class TextSummarizer:
    def __init__(self):
        """
        ملخص النصوص العربية
        """
        # نمط لتقسيم الجمل العربية
        self.sentence_pattern = r'[.!?؟]+'
        
        # كلمات الإيقاف العربية الأساسية
        self.arabic_stopwords = {
            'في', 'من', 'إلى', 'على', 'عن', 'مع', 'بعد', 'قبل', 'عند', 'لدى',
            'هذا', 'هذه', 'ذلك', 'تلك', 'التي', 'الذي', 'التي', 'اللذان', 'اللتان',
            'هو', 'هي', 'أن', 'أنه', 'أنها', 'كان', 'كانت', 'يكون', 'تكون',
            'قد', 'لقد', 'قال', 'قالت', 'يقول', 'تقول', 'أم', 'أما', 'إما',
            'لا', 'لم', 'لن', 'ما', 'ليس', 'غير', 'سوى', 'بل', 'لكن',
            'أو', 'أم', 'حيث', 'كيف', 'متى', 'أين', 'لماذا', 'ماذا', 'مَن',
            'كل', 'جميع', 'بعض', 'معظم', 'أكثر', 'أقل', 'نفس', 'ذات'
        }
    
    def clean_text(self, text):
        """
        تنظيف النص
        
        Args:
            text (str): النص الأصلي
            
        Returns:
            str: النص بعد التنظيف
        """
        # إزالة الرموز الزائدة
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF\s\w\.\!\?\؟]', ' ', text)
        
        # إزالة المسافات الزائدة
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_sentences(self, text):
        """
        تقسيم النص إلى جمل
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة الجمل
        """
        sentences = re.split(self.sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def calculate_sentence_scores(self, sentences, keywords=None):
        """
        حساب درجات الجمل
        
        Args:
            sentences (list): قائمة الجمل
            keywords (list): الكلمات المفتاحية
            
        Returns:
            dict: درجات الجمل
        """
        if not sentences:
            return {}
        
        # حساب TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(self.arabic_stopwords),
                max_features=100,
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # حساب درجة كل جملة بناءً على التشابه
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                # درجة التشابه مع الجمل الأخرى
                similarity_score = np.mean(similarity_matrix[i])
                
                # درجة الطول (الجمل المتوسطة الطول أفضل)
                word_count = len(sentence.split())
                length_score = min(word_count / 20, 1.0) if word_count > 5 else 0.5
                
                # درجة الموقع (الجمل في البداية والنهاية مهمة)
                position_score = 1.0 if i < 3 or i >= len(sentences) - 3 else 0.8
                
                # درجة الكلمات المفتاحية
                keyword_score = 0
                if keywords:
                    sentence_lower = sentence.lower()
                    for keyword in keywords:
                        if keyword.lower() in sentence_lower:
                            keyword_score += 1
                    keyword_score = min(keyword_score / len(keywords), 1.0)
                
                # الدرجة الإجمالية
                total_score = (
                    similarity_score * 0.4 + 
                    length_score * 0.3 + 
                    position_score * 0.2 + 
                    keyword_score * 0.1
                )
                
                sentence_scores[i] = total_score
            
        except Exception:
            # في حالة فشل TF-IDF، استخدم طريقة بسيطة
            sentence_scores = {}
            for i, sentence in enumerate(sentences):
                word_count = len(sentence.split())
                sentence_scores[i] = word_count / 100.0
        
        return sentence_scores
    
    def extract_keywords(self, text, num_keywords=10):
        """
        استخراج الكلمات المفتاحية
        
        Args:
            text (str): النص
            num_keywords (int): عدد الكلمات المفتاحية
            
        Returns:
            list: الكلمات المفتاحية
        """
        # تنظيف النص
        words = re.findall(r'\b\w+\b', text.lower())
        
        # إزالة كلمات الإيقاف والكلمات القصيرة
        filtered_words = [
            word for word in words 
            if word not in self.arabic_stopwords and len(word) > 2
        ]
        
        # حساب التكرارات
        word_freq = Counter(filtered_words)
        
        # أهم الكلمات
        keywords = [word for word, freq in word_freq.most_common(num_keywords)]
        
        return keywords
    
    def summarize(self, text, length="متوسط", extract_keywords=True):
        """
        تلخيص النص
        
        Args:
            text (str): النص المراد تلخيصه
            length (str): طول التلخيص (قصير، متوسط، طويل)
            extract_keywords (bool): استخراج الكلمات المفتاحية
            
        Returns:
            dict: نتائج التلخيص
        """
        if not text.strip():
            return {
                'summary': '',
                'original_words': 0,
                'summary_words': 0,
                'compression_ratio': 0,
                'keywords': []
            }
        
        # تنظيف النص
        cleaned_text = self.clean_text(text)
        
        # تقسيم إلى جمل
        sentences = self.split_sentences(cleaned_text)
        
        if len(sentences) <= 2:
            return {
                'summary': cleaned_text,
                'original_words': len(text.split()),
                'summary_words': len(cleaned_text.split()),
                'compression_ratio': 1.0,
                'keywords': self.extract_keywords(text) if extract_keywords else []
            }
        
        # استخراج الكلمات المفتاحية
        keywords = self.extract_keywords(cleaned_text) if extract_keywords else []
        
        # حساب درجات الجمل
        sentence_scores = self.calculate_sentence_scores(sentences, keywords)
        
        # تحديد عدد الجمل المطلوبة
        length_ratios = {
            "قصير": 0.3,
            "متوسط": 0.5,
            "طويل": 0.7
        }
        
        ratio = length_ratios.get(length, 0.5)
        num_sentences = max(1, int(len(sentences) * ratio))
        num_sentences = min(num_sentences, len(sentences))
        
        # اختيار أفضل الجمل
        top_sentences = heapq.nlargest(
            num_sentences, 
            sentence_scores.items(), 
            key=lambda x: x[1]
        )
        
        # ترتيب الجمل حسب ظهورها في النص الأصلي
        selected_indices = sorted([idx for idx, score in top_sentences])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        # إنشاء التلخيص
        summary = '. '.join(summary_sentences)
        if not summary.endswith('.'):
            summary += '.'
        
        # حساب الإحصائيات
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return {
            'summary': summary,
            'original_words': original_words,
            'summary_words': summary_words,
            'compression_ratio': compression_ratio,
            'keywords': keywords[:8],  # أهم 8 كلمات
            'selected_sentences': len(summary_sentences),
            'total_sentences': len(sentences)
        }
    
    def summarize_batch(self, texts, length="متوسط"):
        """
        تلخيص مجموعة من النصوص
        
        Args:
            texts (list): قائمة النصوص
            length (str): طول التلخيص
            
        Returns:
            list: نتائج التلخيص
        """
        results = []
        for text in texts:
            result = self.summarize(text, length=length)
            results.append(result)
        
        return results