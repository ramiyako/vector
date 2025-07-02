# מדריך ניטור נקיון ספקטוגרמה 🧹

## סקירה כללית

מדריך זה מסביר כיצד לנטר ולוודא שהספקטוגרמות נשארות נקיות וללא רעש, כמו בתמונה השנייה המבוקשת.

## טסטים זמינים

### 1. בדיקה מהירה ⚡

**קובץ**: `test_clean_spectrogram.py`

```bash
python3 test_clean_spectrogram.py
```

**מה זה עושה**:
- יוצר 3 סוגי סיגנלים בדיקה (טון, סוויפ, רב-טון)
- בודק שהדינמיק רנג' ≤35 dB
- יוצר תמונה ויזואלית: `clean_spectrogram_test.png`
- מדווח על בעיות אם קיימות

**זמן ריצה**: ~10 שניות

### 2. בדיקה מקיפה 🔬

**קובץ**: `tests/test_spectrogram_cleanliness.py`

```bash
python3 tests/test_spectrogram_cleanliness.py
```

**מה זה עושה**:
- 6 טסטים מקיפים לנקיון
- בדיקת SNR ובהירות סיגנלים
- בדיקת הפרדת תדרים
- בדיקת דיכוי רעש

**זמן ריצה**: ~30 שניות

### 3. הרצה בתוך הטסט הכללי 📋

**קובץ**: `test_resolution_improvements.py`

```bash
python3 test_resolution_improvements.py
```

הטסט הכללי כולל אוטומטית את בדיקות הנקיון.

## מה לחפש בתוצאות

### ✅ תוצאות תקינות

```
🎯 ✅ QUICK CLEANLINESS CHECK PASSED
Spectrograms are clean and noise-free!

Dynamic range: 30.0 dB (within acceptable range)
Tone burst clarity: 61.3 dB above noise floor
```

### ⚠️ סימני אזהרה

```
⚠️ WARNING: Tone burst dynamic range too high: 45.0 dB
⚠️ CLEANLINESS ISSUES DETECTED
```

**פעולות נדרשות**:
1. בדוק שלא שונו פרמטרי הספקטוגרמה
2. וודא שפונקציית `normalize_spectrogram` עדיין מוגבלת ל-30 dB
3. בדוק שהסינון median עדיין פעיל

## פרמטרים קריטיים לניטור

### בקובץ `utils.py`:

```python
# פונקציית normalize_spectrogram - דינמיק רנג' מוגבל
max_dynamic_range=30  # ✅ צריך להיות ≤35

# חלונות קטנים לתצוגה נקייה
base_window = max(128, min(len(sig) // 8, 1024))  # ✅

# NFFT מוגבל
nfft = max(nfft, 512)  # ✅ לא יותר מ-1024

# סינון median פעיל
Sxx_db = ndimage.median_filter(Sxx_db, size=(2, 1))  # ✅
```

## לוח זמנים מומלץ לניטור

### יומי ⏰
```bash
# בדיקה מהירה (10 שניות)
python3 test_clean_spectrogram.py
```

### שבועי 📅
```bash
# בדיקה מקיפה (30 שניות)
python3 tests/test_spectrogram_cleanliness.py
```

### לפני שחרור גרסה 🚀
```bash
# הרצה מלאה של כל הטסטים
python3 test_resolution_improvements.py
```

## פתרון בעיות נפוצות

### בעיה: דינמיק רנג' גבוה מדי

**תסמינים**: 
```
Dynamic range too high: 45.0 dB (should be ≤35 dB)
```

**פתרון**:
1. בדוק ב-`utils.py` את `normalize_spectrogram`
2. וודא: `max_dynamic_range=30`
3. וודא: `high_percentile=95.0`

### בעיה: סיגנלים לא ברורים

**תסמינים**:
```
Signal too weak: 8.0 dB above noise (should be ≥15 dB)
```

**פתרון**:
1. בדוק שהסינון median פעיל
2. וודא שחלונות FFT לא גדולים מדי
3. בדוק פרמטרי NFFT

### בעיה: ספקטוגרמה ריקה

**תסמינים**: קובץ PNG ריק או שגיאה ביצירה

**פתרון**:
1. הרץ `python3 tests/test_spectrogram_cleanliness.py`
2. בדוק שגיאות בלוג
3. וודא שכל התלויות זמינות

## בדיקה ויזואלית

### קבצי תמונה שנוצרים:

- `clean_spectrogram_test.png` - הבדיקה המהירה
- צריך להיראות כמו **התמונה השנייה** שהראית:
  - בלוקים ברורים של אנרגיה
  - ללא "זמזום" או רעש
  - קווים חדים ומוגדרים

### מה לא צריך להיראות:
- רעש מפוזר בכל הספקטוגרמה
- קווים מטושטשים
- יותר מדי פרטים קטנים
- כמו **התמונה הראשונה** שהראית

## הוספת טסטים חדשים

אם נחוצים טסטים נוספים, הוסף ל-`tests/test_spectrogram_cleanliness.py`:

```python
def test_my_new_feature():
    """Test description"""
    # יצירת סיגנל בדיקה
    # בדיקת תוצאות
    # assert conditions
    return True

# הוסף לרשימה:
tests = [
    test_dynamic_range_limits,
    test_my_new_feature,  # ← כאן
    # ...
]
```

## סיכום

- **הרץ יומי**: `python3 test_clean_spectrogram.py`
- **בדוק ויזואלית**: `clean_spectrogram_test.png`
- **שמור על**: דינמיק רנג' ≤30 dB
- **מטרה**: ספקטוגרמה נקייה כמו התמונה השנייה

🎯 **המטרה**: לוודא שהספקטוגרמה תמיד נותנת **סיגנלים ברורים ונקיים** לניתוח חבילות!