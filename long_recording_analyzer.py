"""
Long Recording Analyzer - מנתח הקלטות ארוכות
מודול לטעינת הקלטות ארוכות (1-2 שניות, 56 MSps), 
זיהוי פקטות באופן אוטומטי, חיתוכן עם מרווח בטיחות,
ובחירת הפקטה האיכותית ביותר מכל סוג.
"""

import os
import numpy as np
import scipy.io as sio
from scipy import signal
from scipy.ndimage import label, find_objects
import matplotlib.pyplot as plt
from collections import defaultdict
import shutil
from datetime import datetime

class LongRecordingAnalyzer:
    def __init__(self, sample_rate=56e6, safety_margin_ms=0.1):
        """
        אתחול מנתח הקלטות ארוכות
        
        Args:
            sample_rate: קצב דגימה (ברירת מחדל 56 MHz)
            safety_margin_ms: מרווח בטיחות בחיתוך הפקטות במילישניות
        """
        self.sample_rate = sample_rate
        self.safety_margin_samples = int(safety_margin_ms * sample_rate / 1000)
        
        # פרמטרים לזיהוי פקטות
        self.power_threshold_db = -40  # סף כוח יחסי לזיהוי פקטות
        self.min_packet_samples = int(0.01 * sample_rate)  # מינימום 10ms לפקטה
        self.max_packet_samples = int(0.5 * sample_rate)   # מקסימום 500ms לפקטה
        
        # פרמטרים לקיבוץ פקטות
        self.frequency_tolerance_hz = 50e3  # סובלנות תדר לקיבוץ פקטות (50 kHz)
        self.bandwidth_tolerance = 0.2  # סובלנות רוחב פס (20%)
        
    def load_recording(self, file_path):
        """
        טעינת הקלטה ארוכה מקובץ MAT
        """
        print(f"📁 טוען הקלטה: {os.path.basename(file_path)}")
        
        try:
            # בדיקת גודל קובץ
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"🗂️ גודל קובץ: {file_size_mb:.1f}MB")
            
            # טעינת הנתונים
            data = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
            
            # מציאת הנתונים הראשיים
            if 'Y' in data:
                recording = data['Y']
            else:
                candidates = [k for k in data.keys() if not k.startswith('__')]
                if candidates:
                    recording = data[candidates[0]]
                else:
                    raise ValueError("לא נמצאו נתונים בקובץ")
            
            # ווידוא שהנתונים הם וקטור
            if recording.ndim > 1:
                recording = recording.flatten()
            
            # המרה לטיפוס יעיל יותר
            recording = recording.astype(np.complex64)
            
            duration_sec = len(recording) / self.sample_rate
            print(f"⏱️ משך הקלטה: {duration_sec:.2f} שניות")
            print(f"📊 מספר דגימות: {len(recording):,}")
            
            return recording
            
        except Exception as e:
            print(f"❌ שגיאה בטעינת הקלטה: {e}")
            raise
    
    def detect_packets(self, recording):
        """
        זיהוי פקטות בהקלטה באופן אוטומטי
        מחזיר רשימה של טיפלים (start_idx, end_idx, power_db, center_freq, bandwidth)
        """
        print("🔍 מזהה פקטות בהקלטה...")
        
        # חישוב עוצמה מיידית
        instant_power = np.abs(recording)**2
        
        # החלקה למניעת רעש
        window_size = max(1, int(0.001 * self.sample_rate))  # 1ms window
        power_smooth = np.convolve(instant_power, np.ones(window_size)/window_size, mode='same')
        
        # המרה ל-dB
        power_db = 10 * np.log10(power_smooth + 1e-12)
        noise_floor_db = np.percentile(power_db, 10)  # רמת רעש בסיס
        
        # זיהוי אזורים מעל הסף
        threshold_db = noise_floor_db + self.power_threshold_db
        above_threshold = power_db > threshold_db
        
        # מציאת אזורים רציפים
        labeled, num_features = label(above_threshold)
        objects = find_objects(labeled)
        
        packets = []
        print(f"📡 נמצאו {num_features} אזורי אות פוטנציאליים")
        
        for i, obj in enumerate(objects):
            if obj[0] is None:
                continue
                
            start_idx = obj[0].start
            end_idx = obj[0].stop
            duration_samples = end_idx - start_idx
            
            # סינון לפי אורך פקטה
            if duration_samples < self.min_packet_samples or duration_samples > self.max_packet_samples:
                continue
            
            # הוספת מרווח בטיחות
            safe_start = max(0, start_idx - self.safety_margin_samples)
            safe_end = min(len(recording), end_idx + self.safety_margin_samples)
            
            # חישוב מאפיינים של הפקטה
            packet_data = recording[safe_start:safe_end]
            
            # חישוב עוצמה ממוצעת
            avg_power_db = np.mean(power_db[start_idx:end_idx])
            
            # חישוב תדר מרכזי ורוחב פס באמצעות FFT
            fft = np.fft.fft(packet_data)
            freqs = np.fft.fftfreq(len(packet_data), 1/self.sample_rate)
            psd = np.abs(fft)**2
            
            # מציאת תדר מרכזי (תדר עם העוצמה המקסימלית)
            peak_idx = np.argmax(psd[:len(psd)//2])  # חצי ראשון של הספקטרום
            center_freq = abs(freqs[peak_idx])
            
            # חישוב רוחב פס (3dB bandwidth)
            max_power = np.max(psd)
            half_power = max_power / 2
            above_half = psd > half_power
            
            if np.any(above_half):
                freq_indices = np.where(above_half)[0]
                bandwidth = (freq_indices[-1] - freq_indices[0]) * self.sample_rate / len(packet_data)
            else:
                bandwidth = self.sample_rate / len(packet_data)  # רזולוציה מינימלית
            
            packets.append({
                'start_idx': safe_start,
                'end_idx': safe_end,
                'original_start': start_idx,
                'original_end': end_idx,
                'power_db': avg_power_db,
                'center_freq': center_freq,
                'bandwidth': bandwidth,
                'duration_ms': (end_idx - start_idx) / self.sample_rate * 1000,
                'snr_db': avg_power_db - noise_floor_db
            })
        
        print(f"✅ זוהו {len(packets)} פקטות תקינות")
        return packets
    
    def group_similar_packets(self, packets):
        """
        קיבוץ פקטות דומות לפי תדר ורוחב פס
        """
        print("🔗 מקבץ פקטות דומות...")
        
        groups = defaultdict(list)
        
        for packet in packets:
            # חיפוש קבוצה מתאימה
            group_key = None
            for existing_key in groups.keys():
                ref_freq, ref_bw = existing_key
                
                # בדיקת קרבה בתדר
                freq_diff = abs(packet['center_freq'] - ref_freq)
                if freq_diff > self.frequency_tolerance_hz:
                    continue
                
                # בדיקת קרבה ברוחב פס
                bw_ratio = abs(packet['bandwidth'] - ref_bw) / max(ref_bw, packet['bandwidth'])
                if bw_ratio > self.bandwidth_tolerance:
                    continue
                
                group_key = existing_key
                break
            
            # יצירת קבוצה חדשה אם לא נמצאה קבוצה מתאימה
            if group_key is None:
                group_key = (packet['center_freq'], packet['bandwidth'])
            
            groups[group_key].append(packet)
        
        print(f"📋 נוצרו {len(groups)} קבוצות פקטות")
        return groups
    
    def select_best_packet(self, packet_group):
        """
        בחירת הפקטה האיכותית ביותר מקבוצה
        """
        if len(packet_group) == 1:
            return packet_group[0]
        
        # ניקוד איכות מורכב
        best_packet = None
        best_score = -float('inf')
        
        for packet in packet_group:
            # ניקוד מבוסס על:
            # 1. SNR (50% מהניקוד)
            # 2. עוצמה (30% מהניקוד)  
            # 3. משך (20% מהניקוד - פקטות ארוכות יותר עדיפות)
            
            snr_score = packet['snr_db'] * 0.5
            power_score = packet['power_db'] * 0.3
            duration_score = packet['duration_ms'] * 0.2
            
            total_score = snr_score + power_score + duration_score
            
            if total_score > best_score:
                best_score = total_score
                best_packet = packet
        
        return best_packet
    
    def extract_packets(self, recording, packets):
        """
        חילוץ הפקטות מההקלטה המקורית
        """
        extracted_packets = []
        
        for packet in packets:
            start_idx = packet['start_idx']
            end_idx = packet['end_idx']
            
            packet_data = recording[start_idx:end_idx]
            extracted_packets.append({
                'data': packet_data,
                'metadata': packet
            })
        
        return extracted_packets
    
    def save_packets(self, extracted_packets, output_dir, base_name="packet"):
        """
        שמירת הפקטות בתיקייה עם אותו פורמט MAT כמו באפליקציה
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        
        for i, packet_info in enumerate(extracted_packets):
            # יצירת שם קובץ עם מידע על התדר ורוחב הפס
            freq_mhz = packet_info['metadata']['center_freq'] / 1e6
            bw_mhz = packet_info['metadata']['bandwidth'] / 1e6
            
            filename = f"{base_name}_{i+1:02d}_freq_{freq_mhz:.1f}MHz_bw_{bw_mhz:.1f}MHz.mat"
            filepath = os.path.join(output_dir, filename)
            
            # שמירה באותו פורמט כמו באפליקציה המקורית
            save_data = {
                'Y': packet_info['data'],
                'metadata': {
                    'center_freq': packet_info['metadata']['center_freq'],
                    'bandwidth': packet_info['metadata']['bandwidth'],
                    'power_db': packet_info['metadata']['power_db'],
                    'snr_db': packet_info['metadata']['snr_db'],
                    'duration_ms': packet_info['metadata']['duration_ms'],
                    'sample_rate': self.sample_rate,
                    'extraction_time': datetime.now().isoformat()
                }
            }
            
            sio.savemat(filepath, save_data)
            saved_files.append(filepath)
            
            print(f"💾 נשמר: {filename} (תדר: {freq_mhz:.1f}MHz, עוצמה: {packet_info['metadata']['power_db']:.1f}dB)")
        
        return saved_files
    
    def analyze_recording(self, input_file, output_dir=None):
        """
        פונקציה ראשית לניתוח הקלטה ארוכה
        """
        print("🚀 מתחיל ניתוח הקלטה ארוכה...")
        
        # יצירת תיקיית פלט אם לא סופקה
        if output_dir is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_dir = f"extracted_packets_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # שלב 1: טעינת ההקלטה
            recording = self.load_recording(input_file)
            
            # שלב 2: זיהוי פקטות
            packets = self.detect_packets(recording)
            
            if not packets:
                print("⚠️ לא נמצאו פקטות בהקלטה")
                return
            
            # שלב 3: קיבוץ פקטות דומות
            groups = self.group_similar_packets(packets)
            
            # שלב 4: בחירת הפקטה הטובה ביותר מכל קבוצה
            best_packets = []
            for group_key, packet_group in groups.items():
                best_packet = self.select_best_packet(packet_group)
                best_packets.append(best_packet)
                
                freq_mhz = group_key[0] / 1e6
                bw_mhz = group_key[1] / 1e6
                print(f"📊 קבוצה תדר {freq_mhz:.1f}MHz: נבחרה פקטה עם SNR {best_packet['snr_db']:.1f}dB מתוך {len(packet_group)} פקטות")
            
            # שלב 5: חילוץ הפקטות הנבחרות
            extracted_packets = self.extract_packets(recording, best_packets)
            
            # שלב 6: שמירה
            saved_files = self.save_packets(extracted_packets, output_dir)
            
            # סיכום
            print(f"\n✅ ניתוח הושלם בהצלחה!")
            print(f"📁 תיקיית פלט: {output_dir}")
            print(f"📦 נשמרו {len(saved_files)} פקטות איכותיות")
            print(f"💽 גודל ממוצע לפקטה: {np.mean([len(ep['data']) for ep in extracted_packets]):.0f} דגימות")
            
            return {
                'output_dir': output_dir,
                'saved_files': saved_files,
                'packet_count': len(saved_files),
                'groups_found': len(groups),
                'total_packets_detected': len(packets)
            }
            
        except Exception as e:
            print(f"❌ שגיאה בניתוח: {e}")
            raise

def main():
    """פונקציה לבדיקה ישירה מהמסוף"""
    import sys
    
    if len(sys.argv) != 2:
        print("שימוש: python long_recording_analyzer.py <קובץ_הקלטה.mat>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"❌ קובץ לא נמצא: {input_file}")
        sys.exit(1)
    
    # יצירת מנתח
    analyzer = LongRecordingAnalyzer()
    
    # ניתוח ההקלטה
    result = analyzer.analyze_recording(input_file)
    
    if result:
        print(f"\n🎯 תוצאות הניתוח:")
        for key, value in result.items():
            print(f"   {key}: {value}")

if __name__ == "__main__":
    main()