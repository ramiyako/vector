"""
Heavy Packet Optimizer
אופטימיזציה מתקדמת לטיפול בפקטות כבדות (1 שניה ב-56MHz = 56 מיליון סמפלים)
"""

import numpy as np
import psutil
import gc
import warnings
from typing import Optional, Tuple, Union
import time

class HeavyPacketOptimizer:
    """אופטימיזציה מתקדמת לפקטות כבדות"""
    
    def __init__(self):
        self.memory_threshold_gb = 4.0  # סף זיכרון בגיגה
        self.chunk_size_mb = 100  # גודל חלק בMB
        self.max_samples_per_chunk = 5_000_000  # מקסימום סמפלים לחלק
        
        # זיהוי זיכרון מערכת
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)
        self.available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        print(f"זיכרון מערכת: {self.total_memory_gb:.1f}GB, זמין: {self.available_memory_gb:.1f}GB")
        
        # התאמת פרמטרים לפי זיכרון זמין
        self._adjust_parameters()
    
    def _adjust_parameters(self):
        """התאמת פרמטרים אוטומטית לפי זיכרון זמין"""
        if self.available_memory_gb < 2:
            self.chunk_size_mb = 50
            self.max_samples_per_chunk = 2_000_000
            print("⚠️ זיכרון נמוך - מצב חסכוני")
        elif self.available_memory_gb > 8:
            self.chunk_size_mb = 200
            self.max_samples_per_chunk = 10_000_000
            print("✅ זיכרון גבוה - מצב מהיר")
        else:
            print("✅ זיכרון רגיל - מצב מאוזן")
    
    def estimate_memory_usage(self, signal_length: int, dtype=np.complex64) -> float:
        """הערכת שימוש בזיכרון במגה בייט"""
        bytes_per_sample = np.dtype(dtype).itemsize
        total_mb = (signal_length * bytes_per_sample) / (1024 * 1024)
        
        # הערכה כוללת עם פעולות עיבוד (x3 בגלל עותקים זמניים)
        estimated_mb = total_mb * 3
        return estimated_mb
    
    def should_use_chunking(self, signal_length: int) -> bool:
        """האם להשתמש בעיבוד מחולק לחלקים"""
        estimated_mb = self.estimate_memory_usage(signal_length)
        return estimated_mb > (self.available_memory_gb * 1024 * 0.5)  # 50% מהזיכרון הזמין
    
    def optimize_data_type(self, signal: np.ndarray) -> np.ndarray:
        """אופטימיזציה של סוג הנתונים"""
        if signal.dtype in [np.complex128, np.float64]:
            # המרה ל-precision נמוך יותר לחיסכון בזיכרון
            if np.iscomplexobj(signal):
                return signal.astype(np.complex64)
            else:
                return signal.astype(np.float32)
        return signal
    
    def chunk_signal(self, signal: np.ndarray, chunk_size: Optional[int] = None) -> list:
        """חלוקת האות לחלקים לעיבוד יעיל"""
        if chunk_size is None:
            chunk_size = self.max_samples_per_chunk
        
        chunks = []
        for i in range(0, len(signal), chunk_size):
            end_idx = min(i + chunk_size, len(signal))
            chunks.append((i, end_idx, signal[i:end_idx]))
        
        print(f"חולק לכ-{len(chunks)} חלקים של עד {chunk_size:,} סמפלים")
        return chunks
    
    def process_heavy_signal(self, signal: np.ndarray, 
                           sample_rate: float,
                           operation: str = "spectrogram",
                           **kwargs) -> tuple:
        """עיבוד אות כבד עם אופטימיזציה מתקדמת"""
        
        print(f"מעבד אות כבד: {len(signal):,} סמפלים ({len(signal)/sample_rate:.2f} שניות)")
        
        # אופטימיזציה ראשונית
        signal = self.optimize_data_type(signal)
        
        # בדיקה האם נדרש עיבוד מחולק
        if self.should_use_chunking(len(signal)):
            print("🔄 משתמש בעיבוד מחולק לחלקים")
            return self._process_chunked(signal, sample_rate, operation, **kwargs)
        else:
            print("⚡ עיבוד ישיר")
            return self._process_direct(signal, sample_rate, operation, **kwargs)
    
    def _process_direct(self, signal: np.ndarray, sample_rate: float, 
                       operation: str, **kwargs) -> tuple:
        """עיבוד ישיר לאותות קטנים יותר"""
        start_time = time.time()
        
        if operation == "spectrogram":
            result = self._create_optimized_spectrogram(signal, sample_rate, **kwargs)
        else:
            raise ValueError(f"פעולה לא נתמכת: {operation}")
        
        process_time = time.time() - start_time
        print(f"⏱️ זמן עיבוד: {process_time:.2f} שניות")
        
        return result
    
    def _process_chunked(self, signal: np.ndarray, sample_rate: float,
                        operation: str, **kwargs) -> tuple:
        """עיבוד מחולק לחלקים"""
        start_time = time.time()
        
        # חלוקה לחלקים עם חפיפה
        overlap_samples = int(sample_rate * 0.01)  # חפיפה של 10ms
        chunk_size = self.max_samples_per_chunk
        
        chunks_results = []
        total_chunks = (len(signal) + chunk_size - 1) // chunk_size
        
        for i in range(0, len(signal), chunk_size - overlap_samples):
            chunk_num = i // (chunk_size - overlap_samples) + 1
            end_idx = min(i + chunk_size, len(signal))
            chunk = signal[i:end_idx]
            
            print(f"מעבד חלק {chunk_num}/{total_chunks} ({len(chunk):,} סמפלים)")
            
            # עיבוד החלק
            if operation == "spectrogram":
                result = self._create_optimized_spectrogram(chunk, sample_rate, **kwargs)
                chunks_results.append((i, result))
            
            # ניקוי זיכרון
            del chunk
            gc.collect()
            
            if end_idx >= len(signal):
                break
        
        # איחוד תוצאות
        print("🔗 מאחד תוצאות...")
        final_result = self._merge_chunks_results(chunks_results, operation)
        
        process_time = time.time() - start_time
        print(f"⏱️ זמן עיבוד כולל: {process_time:.2f} שניות")
        
        return final_result
    
    def _create_optimized_spectrogram(self, signal: np.ndarray, sample_rate: float,
                                    max_samples: int = 2_000_000,
                                    time_resolution_us: float = 10.0,
                                    adaptive_resolution: bool = True,
                                    **kwargs) -> tuple:
        """יצירת ספקטרוגרמה מותאמת לפקטות כבדות"""
        
        from utils import create_spectrogram
        
        # התאמת פרמטרים לאות כבד
        if len(signal) > max_samples:
            # דגימה מחדש אגרסיבית יותר
            downsample_factor = int(np.ceil(len(signal) / max_samples))
            signal_ds = signal[::downsample_factor]
            fs_ds = sample_rate / downsample_factor
            
            print(f"דגימה מחדש: גורם {downsample_factor}, אורך חדש: {len(signal_ds):,}")
        else:
            signal_ds = signal
            fs_ds = sample_rate
        
        # התאמת רזולוציית זמן לאות כבד
        signal_duration_ms = len(signal_ds) / fs_ds * 1000
        if signal_duration_ms > 1000:  # מעל שנייה
            time_resolution_us = min(time_resolution_us, 50.0)  # לא פחות מ-50μs
        
        return create_spectrogram(
            signal_ds, fs_ds,
            max_samples=max_samples,
            time_resolution_us=int(time_resolution_us),
            adaptive_resolution=adaptive_resolution,
            **kwargs
        )
    
    def _merge_chunks_results(self, chunks_results: list, operation: str) -> tuple:
        """איחוד תוצאות מחלקים"""
        if operation == "spectrogram":
            return self._merge_spectrograms(chunks_results)
        else:
            raise ValueError(f"איחוד לא נתמך עבור: {operation}")
    
    def _merge_spectrograms(self, chunks_results: list) -> tuple:
        """איחוד ספקטרוגרמות מחלקים"""
        if not chunks_results:
            raise ValueError("אין תוצאות לאיחוד")
        
        # השוואת תדרים (צריכים להיות זהים)
        first_freqs = chunks_results[0][1][0]
        
        # איחוד זמנים וספקטרוגרמות
        all_times = []
        all_spectrograms = []
        
        time_offset = 0
        for chunk_idx, (start_sample, (freqs, times, Sxx)) in enumerate(chunks_results):
            # התאמת זמנים
            adjusted_times = times + time_offset
            all_times.append(adjusted_times)
            all_spectrograms.append(Sxx)
            
            # עדכון offset לחלק הבא
            if len(adjusted_times) > 0:
                time_offset = adjusted_times[-1] + (adjusted_times[1] - adjusted_times[0] if len(adjusted_times) > 1 else 0)
        
        # איחוד סופי
        merged_times = np.concatenate(all_times)
        merged_spectrogram = np.concatenate(all_spectrograms, axis=1)
        
        return first_freqs, merged_times, merged_spectrogram
    
    def monitor_memory_usage(self) -> dict:
        """ניטור שימוש בזיכרון"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent
        }
    
    def cleanup_memory(self):
        """ניקוי זיכרון אגרסיבי"""
        gc.collect()
        memory_after = self.monitor_memory_usage()
        print(f"ניקוי זיכרון - זמין: {memory_after['available_gb']:.1f}GB")


# פונקציות עזר גלובליות
def optimize_for_heavy_packets():
    """אופטימיזציה גלובלית למערכת לפקטות כבדות"""
    
    # הגדרות numpy מותאמות
    try:
        import os
        # הגבלת מספר threads של numpy לחיסכון בזיכרון
        os.environ['OPENBLAS_NUM_THREADS'] = '2'
        os.environ['MKL_NUM_THREADS'] = '2'
        os.environ['OMP_NUM_THREADS'] = '2'
        
        # אופטימיזציות זיכרון
        np.seterr(all='ignore')  # התעלמות מ-warnings לביצועים
        
        print("✅ אופטימיזציות גלובליות הופעלו")
    except Exception as e:
        print(f"⚠️ שגיאה באופטימיזציות: {e}")


def estimate_processing_time(signal_length: int, sample_rate: float) -> dict:
    """הערכת זמני עיבוד לפקטות כבדות"""
    
    optimizer = HeavyPacketOptimizer()
    
    # הערכות זמן בהתבסס על גודל ואיכות מערכת
    base_time_per_sample = 2e-7  # זמן בסיס לסמפל (שניות)
    memory_factor = 1.0
    
    if optimizer.should_use_chunking(signal_length):
        memory_factor = 1.5  # עיבוד מחולק לוקח זמן נוסף
    
    estimated_seconds = signal_length * base_time_per_sample * memory_factor
    
    return {
        'signal_length': signal_length,
        'signal_duration_sec': signal_length / sample_rate,
        'estimated_processing_sec': estimated_seconds,
        'needs_chunking': optimizer.should_use_chunking(signal_length),
        'estimated_memory_mb': optimizer.estimate_memory_usage(signal_length)
    }


# הפעלה אוטומטית של אופטימיזציות
optimize_for_heavy_packets()