import re
from collections import defaultdict, Counter
import datetime

class NERExtractor:
    def __init__(self):
        """
        مستخرج الكيانات والأحداث من النصوص العربية
        """
        
        # أنماط الأشخاص
        self.person_patterns = [
            r'\b(?:الدكتور|الأستاذ|المهندس|الوزير|الرئيس|الملك|الأمير|السيد|السيدة)\s+([^\s]+(?:\s+[^\s]+)*)',
            r'\b([أ-ي]{2,})\s+(?:قال|قالت|أعلن|أعلنت|أكد|أكدت|صرح|صرحت)',
            r'\b(?:محمد|أحمد|علي|حسن|خالد|عبد\s*الله|عبد\s*الرحمن|فاطمة|عائشة|خديجة|مريم)\s+([أ-ي]+)',
        ]
        
        # أنماط الأماكن
        self.location_patterns = [
            r'\b(?:في|إلى|من|بـ|عبر)\s+((?:[أ-ي]+\s*){1,3})(?:\s|،|\.)',
            r'\b(القاهرة|الإسكندرية|الرياض|جدة|دبي|أبو\s*ظبي|بيروت|دمشق|عمان|الدوحة|الكويت|المنامة|مسقط|صنعاء|بغداد|تونس|الجزائر|الرباط|طرابلس)\b',
            r'\b(مصر|السعودية|الإمارات|لبنان|سوريا|الأردن|قطر|الكويت|البحرين|عمان|اليمن|العراق|تونس|الجزائر|المغرب|ليبيا|فلسطين|السودان)\b',
            r'\b(?:محافظة|ولاية|إمارة|منطقة|حي|شارع|ميدان)\s+([أ-ي\s]+)',
        ]
        
        # أنماط المؤسسات
        self.organization_patterns = [
            r'\b(?:شركة|مؤسسة|هيئة|مجلس|لجنة|منظمة|اتحاد|حزب|جامعة|كلية|مستشفى|مصرف|بنك)\s+([أ-ي\s]+)',
            r'\b(?:وزارة)\s+([أ-ي\s]+)',
            r'\b(?:الأمم\s*المتحدة|جامعة\s*الدول\s*العربية|مجلس\s*التعاون\s*الخليجي|الاتحاد\s*الأوروبي)\b',
        ]
        
        # أنماط التواريخ
        self.date_patterns = [
            r'\b(?:يوم|أمس|اليوم|غدا|مساء)\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\s+\d{4}\b',
            r'\b\d{1,2}\s+(?:يناير|فبراير|مارس|أبريل|مايو|يونيو|يوليو|أغسطس|سبتمبر|أكتوبر|نوفمبر|ديسمبر)\b',
        ]
        
        # أنماط الأحداث
        self.event_patterns = [
            r'(?:اجتماع|مؤتمر|ندوة|ورشة|دورة|بطولة|مهرجان|احتفال|حفل)\s+([أ-ي\s]+)',
            r'(?:انطلق|بدأ|اختتم|انتهى|عقد|أقيم)\s+([أ-ي\s]+)',
            r'(?:زيارة|لقاء|استقبال)\s+([أ-ي\s]+)',
        ]
        
        # كلمات الأفعال المهمة للأحداث
        self.event_verbs = [
            'التقى', 'اجتمع', 'ناقش', 'بحث', 'أعلن', 'وقع', 'افتتح', 'اختتم',
            'زار', 'استقبل', 'سافر', 'وصل', 'غادر', 'عاد', 'شارك', 'حضر'
        ]
    
    def extract_persons(self, text):
        """
        استخراج أسماء الأشخاص
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة الأشخاص
        """
        persons = []
        
        for pattern in self.person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                person = match.group(1).strip()
                if len(person.split()) <= 4 and len(person) > 2:  # تجنب النتائج الطويلة جداً
                    persons.append({
                        'text': person,
                        'type': 'شخص',
                        'confidence': 0.8
                    })
        
        return persons
    
    def extract_locations(self, text):
        """
        استخراج الأماكن
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة الأماكن
        """
        locations = []
        
        for pattern in self.location_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                location = match.group(1).strip()
                if len(location.split()) <= 3 and len(location) > 2:
                    confidence = 0.9 if any(city in location for city in ['القاهرة', 'الرياض', 'دبي']) else 0.7
                    locations.append({
                        'text': location,
                        'type': 'مكان',
                        'confidence': confidence
                    })
        
        return locations
    
    def extract_organizations(self, text):
        """
        استخراج المؤسسات
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة المؤسسات
        """
        organizations = []
        
        for pattern in self.organization_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    org = match.group(1).strip()
                    if len(org.split()) <= 4 and len(org) > 2:
                        organizations.append({
                            'text': org,
                            'type': 'مؤسسة',
                            'confidence': 0.8
                        })
                else:
                    org = match.group(0).strip()
                    organizations.append({
                        'text': org,
                        'type': 'مؤسسة',
                        'confidence': 0.9
                    })
        
        return organizations
    
    def extract_dates(self, text):
        """
        استخراج التواريخ
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة التواريخ
        """
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group(0).strip()
                dates.append({
                    'text': date_text,
                    'type': 'تاريخ',
                    'confidence': 0.9
                })
        
        return dates
    
    def extract_entities(self, text):
        """
        استخراج جميع الكيانات
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة جميع الكيانات
        """
        if not text.strip():
            return []
        
        entities = []
        
        # استخراج الأشخاص
        entities.extend(self.extract_persons(text))
        
        # استخراج الأماكن
        entities.extend(self.extract_locations(text))
        
        # استخراج المؤسسات
        entities.extend(self.extract_organizations(text))
        
        # استخراج التواريخ
        entities.extend(self.extract_dates(text))
        
        # إزالة التكرارات
        unique_entities = []
        seen = set()
        
        for entity in entities:
            entity_key = (entity['text'].lower(), entity['type'])
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def extract_events(self, text):
        """
        استخراج الأحداث
        
        Args:
            text (str): النص
            
        Returns:
            list: قائمة الأحداث
        """
        events = []
        sentences = re.split(r'[.!?؟]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # البحث عن أنماط الأحداث
            for pattern in self.event_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    event_text = match.group(0).strip()
                    
                    # استخراج المعلومات الإضافية
                    participants = []
                    location = None
                    time = None
                    
                    # البحث عن المشاركين
                    persons = self.extract_persons(sentence)
                    participants = [p['text'] for p in persons]
                    
                    # البحث عن المكان
                    locations = self.extract_locations(sentence)
                    if locations:
                        location = locations[0]['text']
                    
                    # البحث عن الوقت
                    dates = self.extract_dates(sentence)
                    if dates:
                        time = dates[0]['text']
                    
                    events.append({
                        'event': event_text,
                        'participants': participants,
                        'location': location,
                        'time': time,
                        'sentence': sentence,
                        'confidence': 0.7
                    })
            
            # البحث عن الأفعال المهمة
            for verb in self.event_verbs:
                if verb in sentence:
                    # استخراج السياق
                    verb_index = sentence.find(verb)
                    context = sentence[max(0, verb_index-50):verb_index+100]
                    
                    # استخراج المعلومات
                    persons = self.extract_persons(context)
                    locations = self.extract_locations(context)
                    dates = self.extract_dates(context)
                    
                    events.append({
                        'event': f'{verb} - {context[:50]}...',
                        'participants': [p['text'] for p in persons],
                        'location': locations[0]['text'] if locations else None,
                        'time': dates[0]['text'] if dates else None,
                        'sentence': sentence,
                        'confidence': 0.6
                    })
        
        # إزالة التكرارات وترتيب حسب الثقة
        unique_events = []
        seen_events = set()
        
        for event in events:
            event_key = event['event'][:30].lower()
            if event_key not in seen_events:
                seen_events.add(event_key)
                unique_events.append(event)
        
        # ترتيب حسب الثقة
        unique_events.sort(key=lambda x: x['confidence'], reverse=True)
        
        return unique_events[:10]  # أهم 10 أحداث
    
    def analyze_text(self, text):
        """
        تحليل شامل للنص
        
        Args:
            text (str): النص
            
        Returns:
            dict: نتائج التحليل الشامل
        """
        entities = self.extract_entities(text)
        events = self.extract_events(text)
        
        # إحصائيات الكيانات
        entity_stats = Counter(entity['type'] for entity in entities)
        
        # أهم الكيانات
        top_entities = {}
        for entity_type in ['شخص', 'مكان', 'مؤسسة', 'تاريخ']:
            type_entities = [e for e in entities if e['type'] == entity_type]
            top_entities[entity_type] = sorted(
                type_entities, 
                key=lambda x: x['confidence'], 
                reverse=True
            )[:5]
        
        return {
            'entities': entities,
            'events': events,
            'entity_statistics': dict(entity_stats),
            'top_entities': top_entities,
            'total_entities': len(entities),
            'total_events': len(events)
        }