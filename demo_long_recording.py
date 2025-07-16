#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo script for Long Recording Analyzer
סקריפט דמו למערכת ניתוח הקלטות ארוכות
"""

import numpy as np
import scipy.io as sio
import os
from long_recording_analyzer import LongRecordingAnalyzer

def create_demo_recording():
    """יצירת הקלטת דמו עם מספר פקטות לבדיקה"""
    
    sample_rate = 56e6  # 56 MHz
    duration = 1.5  # 1.5 seconds
    total_samples = int(duration * sample_rate)
    
    print(f"🔨 יוצר הקלטת דמו: {duration} שניות, {sample_rate/1e6:.0f} MHz")
    
    # יצירת רעש בסיס
    noise_level = 0.01
    recording = noise_level * (np.random.randn(total_samples) + 1j * np.random.randn(total_samples))
    
    # הוספת פקטות בתדרים שונים
    t = np.arange(total_samples) / sample_rate
    
    # פקטה 1: 5 MHz, באמצע ההקלטה
    packet1_start = int(0.3 * sample_rate)
    packet1_end = int(0.32 * sample_rate)
    packet1_freq = 5e6
    packet1_amplitude = 0.3
    packet1_signal = packet1_amplitude * np.exp(1j * 2 * np.pi * packet1_freq * t[packet1_start:packet1_end])
    recording[packet1_start:packet1_end] += packet1_signal
    
    # פקטה 2: 10 MHz, מוקדם יותר
    packet2_start = int(0.1 * sample_rate)
    packet2_end = int(0.13 * sample_rate)
    packet2_freq = 10e6
    packet2_amplitude = 0.25
    packet2_signal = packet2_amplitude * np.exp(1j * 2 * np.pi * packet2_freq * t[packet2_start:packet2_end])
    recording[packet2_start:packet2_end] += packet2_signal
    
    # פקטה 3: 5 MHz שוב (צריכה להיכלל באותה קבוצה)
    packet3_start = int(0.7 * sample_rate)
    packet3_end = int(0.72 * sample_rate)
    packet3_freq = 5e6
    packet3_amplitude = 0.2  # יותר חלשה מהפקטה הראשונה
    packet3_signal = packet3_amplitude * np.exp(1j * 2 * np.pi * packet3_freq * t[packet3_start:packet3_end])
    recording[packet3_start:packet3_end] += packet3_signal
    
    # פקטה 4: 15 MHz, מאוחר יותר
    packet4_start = int(1.0 * sample_rate)
    packet4_end = int(1.025 * sample_rate)
    packet4_freq = 15e6
    packet4_amplitude = 0.35
    packet4_signal = packet4_amplitude * np.exp(1j * 2 * np.pi * packet4_freq * t[packet4_start:packet4_end])
    recording[packet4_start:packet4_end] += packet4_signal
    
    # המרה לטיפוס יעיל
    recording = recording.astype(np.complex64)
    
    print(f"✅ נוצרו 4 פקטות:")
    print(f"   פקטה 1: תדר {packet1_freq/1e6:.0f}MHz, עוצמה {packet1_amplitude:.2f}")
    print(f"   פקטה 2: תדר {packet2_freq/1e6:.0f}MHz, עוצמה {packet2_amplitude:.2f}")
    print(f"   פקטה 3: תדר {packet3_freq/1e6:.0f}MHz, עוצמה {packet3_amplitude:.2f} (דומה לפקטה 1)")
    print(f"   פקטה 4: תדר {packet4_freq/1e6:.0f}MHz, עוצמה {packet4_amplitude:.2f}")
    
    return recording, sample_rate

def save_demo_recording(recording, sample_rate, filename="demo_long_recording.mat"):
    """שמירת הקלטת הדמו"""
    
    save_data = {
        'Y': recording,
        'sample_rate': sample_rate,
        'description': 'Demo long recording with 4 packets at different frequencies'
    }
    
    sio.savemat(filename, save_data)
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"💾 נשמרה הקלטת דמו: {filename} ({file_size_mb:.1f}MB)")
    
    return filename

def main():
    """פונקציה ראשית להדגמה"""
    
    print("🚀 הדגמת מערכת ניתוח הקלטות ארוכות")
    print("="*50)
    
    # יצירת הקלטת דמו
    recording, sample_rate = create_demo_recording()
    demo_file = save_demo_recording(recording, sample_rate)
    
    print("\n🔍 מתחיל ניתוח ההקלטה...")
    print("="*50)
    
    # יצירת מנתח
    analyzer = LongRecordingAnalyzer(
        sample_rate=sample_rate,
        safety_margin_ms=0.1
    )
    
    # ניתוח ההקלטה
    try:
        result = analyzer.analyze_recording(demo_file, "demo_extracted_packets")
        
        if result:
            print("\n🎯 סיכום הדגמה:")
            print("="*50)
            print(f"📁 תיקיית פלט: {result['output_dir']}")
            print(f"📦 פקטות שנשמרו: {result['packet_count']}")
            print(f"🔍 קבוצות שזוהו: {result['groups_found']}")
            print(f"📡 סה״כ פקטות שנמצאו: {result['total_packets_detected']}")
            
            print(f"\n✅ הדגמה הושלמה בהצלחה!")
            print(f"   הפקטות החולצות נמצאות בתיקייה: {result['output_dir']}")
            print(f"   ניתן לטעון אותן באפליקציה הראשית לעבודה נוספת")
            
        else:
            print("⚠️ לא נמצאו פקטות בהקלטת הדמו")
            
    except Exception as e:
        print(f"❌ שגיאה בהדגמה: {e}")
        
    finally:
        # ניקוי קובץ הדמו
        if os.path.exists(demo_file):
            os.remove(demo_file)
            print(f"🧹 נוקה קובץ הדמו: {demo_file}")

if __name__ == "__main__":
    main()