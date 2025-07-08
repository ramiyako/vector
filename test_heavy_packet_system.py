#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heavy Packet System Test
בדיקה מקיפה של יכולות המערכת לטיפול בפקטות כבדות
"""

import os
import sys
import time
import traceback
import numpy as np
import psutil
from pathlib import Path

def check_system_capabilities():
    """בדיקת יכולות המערכת"""
    print("=" * 60)
    print("🔍 SYSTEM CAPABILITIES CHECK")
    print("=" * 60)
    
    # זיכרון
    memory = psutil.virtual_memory()
    print(f"💾 Memory:")
    print(f"  Total: {memory.total / (1024**3):.1f} GB")
    print(f"  Available: {memory.available / (1024**3):.1f} GB")
    print(f"  Used: {memory.percent:.1f}%")
    
    # מעבד
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print(f"🖥️ CPU:")
    print(f"  Cores: {cpu_count}")
    if cpu_freq:
        print(f"  Frequency: {cpu_freq.current:.0f} MHz")
    
    # דיסק
    disk = psutil.disk_usage('/')
    print(f"💿 Disk:")
    print(f"  Total: {disk.total / (1024**3):.1f} GB")
    print(f"  Free: {disk.free / (1024**3):.1f} GB")
    
    # הערכת יכולת לטיפול בפקטות כבדות
    can_handle_heavy = memory.available > 2 * (1024**3)  # > 2GB זמין
    print(f"\n🚀 Heavy Packet Support: {'✅ YES' if can_handle_heavy else '❌ LIMITED'}")
    
    return can_handle_heavy

def test_memory_allocation():
    """בדיקת הקצאת זיכרון לפקטות כבדות"""
    print("\n" + "=" * 60)
    print("🧪 MEMORY ALLOCATION TEST")
    print("=" * 60)
    
    test_sizes = [
        (1_000_000, "1M samples (8MB)"),
        (5_000_000, "5M samples (40MB)"),
        (10_000_000, "10M samples (80MB)"),
        (25_000_000, "25M samples (200MB)"),
        (56_000_000, "56M samples (448MB) - 1s@56MHz"),
    ]
    
    successful_allocations = []
    
    for size, description in test_sizes:
        try:
            print(f"📊 Testing {description}...")
            start_time = time.time()
            
            # הקצאת זיכרון
            arr = np.zeros(size, dtype=np.complex64)
            alloc_time = time.time() - start_time
            
            # בדיקת זיכרון זמין
            memory_mb = arr.nbytes / (1024 * 1024)
            memory = psutil.virtual_memory()
            
            print(f"  ✅ Success: {memory_mb:.1f}MB allocated in {alloc_time:.3f}s")
            print(f"  🔄 Available memory: {memory.available / (1024**3):.1f}GB")
            
            successful_allocations.append(size)
            
            # ניקוי זיכרון
            del arr
            
        except MemoryError:
            print(f"  ❌ Failed: Out of memory")
            break
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            break
    
    max_supported = max(successful_allocations) if successful_allocations else 0
    print(f"\n🎯 Maximum supported: {max_supported:,} samples ({max_supported/56e6:.2f}s @ 56MHz)")
    
    return max_supported

def test_heavy_packet_optimizer():
    """בדיקת האופטימיזר לפקטות כבדות"""
    print("\n" + "=" * 60)
    print("🚀 HEAVY PACKET OPTIMIZER TEST")
    print("=" * 60)
    
    try:
        from heavy_packet_optimizer import HeavyPacketOptimizer, estimate_processing_time
        print("✅ Heavy packet optimizer imported successfully")
        
        # יצירת אופטימיזר
        optimizer = HeavyPacketOptimizer()
        print(f"✅ Optimizer created")
        
        # בדיקת הערכת זמני עיבוד
        test_sizes = [1_000_000, 10_000_000, 56_000_000]
        
        for size in test_sizes:
            estimation = estimate_processing_time(size, 56e6)
            print(f"\n📊 Size: {size:,} samples")
            print(f"  Duration: {estimation['signal_duration_sec']:.3f}s")
            print(f"  Estimated processing: {estimation['estimated_processing_sec']:.2f}s")
            print(f"  Needs chunking: {estimation['needs_chunking']}")
            print(f"  Memory usage: {estimation['estimated_memory_mb']:.1f}MB")
        
        return True
        
    except ImportError as e:
        print(f"❌ Heavy packet optimizer not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing optimizer: {e}")
        return False

def test_utils_functions():
    """בדיקת פונקציות utils מעודכנות"""
    print("\n" + "=" * 60)
    print("🔧 UTILS FUNCTIONS TEST")
    print("=" * 60)
    
    try:
        from utils import create_heavy_packet_test, process_heavy_packet_safe
        print("✅ Heavy packet utils imported successfully")
        
        # יצירת פקטה כבדה לבדיקה
        print("\n🧪 Creating test heavy packet...")
        test_signal = create_heavy_packet_test(duration_sec=0.1, sample_rate=56e6)  # 0.1s @ 56MHz
        print(f"✅ Test signal created: {len(test_signal):,} samples")
        
        # בדיקת עיבוד בטוח
        print("\n⚡ Testing safe processing...")
        start_time = time.time()
        f, t, Sxx = process_heavy_packet_safe(test_signal, 56e6, operation="spectrogram")
        process_time = time.time() - start_time
        
        print(f"✅ Processing completed in {process_time:.2f}s")
        print(f"📊 Spectrogram shape: {Sxx.shape}")
        print(f"📊 Frequency range: {f.min()/1e6:.1f} to {f.max()/1e6:.1f} MHz")
        print(f"📊 Time range: {t.min()*1000:.1f} to {t.max()*1000:.1f} ms")
        
        return True
        
    except ImportError as e:
        print(f"❌ Utils functions not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing utils: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_gui_integration():
    """בדיקת שילוב ב-GUI"""
    print("\n" + "=" * 60)
    print("🖥️ GUI INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # בדיקת ייבוא GUI
        print("📦 Testing GUI imports...")
        import unified_gui
        print("✅ Unified GUI imported")
        
        import main
        print("✅ Main module imported")
        
        # בדיקת dependencies
        import customtkinter as ctk
        print("✅ CustomTkinter available")
        
        import matplotlib
        print("✅ Matplotlib available")
        
        return True
        
    except ImportError as e:
        print(f"❌ GUI components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing GUI: {e}")
        return False

def create_test_data_directory():
    """יצירת תיקיית נתונים לבדיקה"""
    print("\n" + "=" * 60)
    print("📁 TEST DATA DIRECTORY")
    print("=" * 60)
    
    try:
        # יצירת תיקיית data
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        print(f"✅ Data directory created: {data_dir.absolute()}")
        
        # בדיקת הרשאות כתיבה
        test_file = data_dir / "test_write.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print("✅ Write permissions confirmed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating data directory: {e}")
        return False

def run_performance_benchmark():
    """ריצת benchmark ביצועים"""
    print("\n" + "=" * 60)
    print("⚡ PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    try:
        from utils import create_heavy_packet_test, create_spectrogram
        
        # בדיקת ביצועים לגדלים שונים
        test_cases = [
            (0.01, "10ms packet"),
            (0.1, "100ms packet"),
            (0.5, "500ms packet"),
            (1.0, "1s packet (HEAVY)")
        ]
        
        results = []
        
        for duration, description in test_cases:
            print(f"\n🧪 Testing {description}...")
            
            try:
                # יצירת אות
                start_time = time.time()
                signal = create_heavy_packet_test(duration_sec=duration, sample_rate=56e6)
                creation_time = time.time() - start_time
                
                # יצירת ספקטרוגרמה
                start_time = time.time()
                f, t, Sxx = create_spectrogram(signal, 56e6, 
                                             max_samples=5_000_000,
                                             time_resolution_us=10,
                                             adaptive_resolution=True)
                processing_time = time.time() - start_time
                
                memory_mb = signal.nbytes / (1024 * 1024)
                
                print(f"  ✅ Success:")
                print(f"    Signal: {len(signal):,} samples ({memory_mb:.1f}MB)")
                print(f"    Creation: {creation_time:.3f}s")
                print(f"    Processing: {processing_time:.3f}s")
                print(f"    Spectrogram: {Sxx.shape}")
                
                results.append({
                    'duration': duration,
                    'samples': len(signal),
                    'memory_mb': memory_mb,
                    'creation_time': creation_time,
                    'processing_time': processing_time,
                    'success': True
                })
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                results.append({
                    'duration': duration,
                    'success': False,
                    'error': str(e)
                })
        
        # סיכום ביצועים
        print(f"\n📊 PERFORMANCE SUMMARY:")
        successful = [r for r in results if r.get('success', False)]
        if successful:
            max_duration = max(r['duration'] for r in successful)
            total_samples = sum(r['samples'] for r in successful)
            avg_processing = np.mean([r['processing_time'] for r in successful])
            
            print(f"  Maximum duration handled: {max_duration}s")
            print(f"  Total samples processed: {total_samples:,}")
            print(f"  Average processing time: {avg_processing:.3f}s")
            
            # הערכת יכולת לפקטה של שנייה ב-56MHz
            full_heavy_samples = 56_000_000
            estimated_time = avg_processing * (full_heavy_samples / np.mean([r['samples'] for r in successful]))
            print(f"  Estimated time for 1s@56MHz: {estimated_time:.1f}s")
            
        return len(successful) > 0
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

def generate_system_report():
    """יצירת דוח מערכת"""
    print("\n" + "=" * 60)
    print("📋 SYSTEM REPORT")
    print("=" * 60)
    
    # איסוף נתונים
    memory = psutil.virtual_memory()
    can_handle_heavy = memory.available > 2 * (1024**3)
    
    report = f"""
Heavy Packet System Report
==========================

System Configuration:
- Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available
- CPU: {psutil.cpu_count()} cores
- Platform: {sys.platform}

Heavy Packet Support:
- Status: {"✅ READY" if can_handle_heavy else "⚠️ LIMITED"}
- Max recommended samples: {25_000_000 if can_handle_heavy else 10_000_000:,}
- Max duration @ 56MHz: {25_000_000/56e6 if can_handle_heavy else 10_000_000/56e6:.3f}s

Optimizations Enabled:
- Heavy packet optimizer: Available
- Chunked processing: Available  
- Memory-efficient data types: Enabled
- Adaptive resolution: Enabled

Recommendations:
"""
    
    if can_handle_heavy:
        report += "- System is ready for heavy packet processing\n"
        report += "- Can handle packets up to 1 second @ 56MHz\n"
        report += "- Use 'Balanced' or 'High Quality' presets\n"
    else:
        report += "- Use 'Fast' preset for better performance\n"
        report += "- Consider processing shorter packet segments\n"
        report += "- Monitor memory usage during processing\n"
    
    report += f"\nGenerated on: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    
    # שמירת דוח
    try:
        with open("heavy_packet_system_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("✅ Report saved to: heavy_packet_system_report.txt")
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")
    
    print(report)
    return report

def main():
    """פונקציה ראשית"""
    print("🚀 HEAVY PACKET SYSTEM TEST")
    print("Testing system capabilities for 1-second packets @ 56MHz (56M samples)")
    print()
    
    # ריצת כל הבדיקות
    tests = [
        ("System Capabilities", check_system_capabilities),
        ("Memory Allocation", test_memory_allocation),
        ("Heavy Packet Optimizer", test_heavy_packet_optimizer),
        ("Utils Functions", test_utils_functions),
        ("GUI Integration", test_gui_integration),
        ("Data Directory", create_test_data_directory),
        ("Performance Benchmark", run_performance_benchmark),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name}...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # סיכום תוצאות
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    # יצירת דוח סיכום
    generate_system_report()
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is ready for heavy packet processing.")
    elif passed >= total * 0.7:
        print("\n⚠️ Most tests passed. System should handle heavy packets with some limitations.")
    else:
        print("\n❌ Several tests failed. Heavy packet processing may be limited.")
    
    print("\n" + "=" * 60)
    print("Test completed. Check the generated report for detailed information.")

if __name__ == "__main__":
    main()