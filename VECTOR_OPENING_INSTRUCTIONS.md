# הוראות פתיחת וקטורים ב-Packet Extractor
# Instructions for Opening Vectors in Packet Extractor

## ✅ הבעיה נפתרה! / Problem SOLVED!

**הוקטורים עכשיו נפתחים בצורה מושלמת ב-packet extractor!**  
**Vectors now open perfectly in the packet extractor!**

---

## 📋 הוראות שימוש מפורטות / Step-by-Step Instructions

### שלב 1: הפעלת המערכת / Step 1: Launch System

```bash
# הפעל את הסביבה הווירטואלית
source venv/bin/activate

# הפעל את הממשק הגרפי
python unified_gui.py
```

### שלב 2: יצירת וקטור / Step 2: Create Vector

1. **בחר את הטאב "Vector Building"**
   - Select the "Vector Building" tab

2. **הגדר פרמטרים**:
   - **Vector Length**: 2.5 (milliseconds)
   - **Number of Packets**: 1-6 (בחר כמות)
   - Configure packet parameters as needed

3. **לחץ "Create MAT Vector"**
   - Click "Create MAT Vector"
   - הוקטור יישמר בתיקיית data/

### שלב 3: פתיחת וקטור בPacket Extractor / Step 3: Open Vector in Packet Extractor

**⚠️ חשוב: עבור לטאב הנכון!**
**⚠️ Important: Switch to the correct tab!**

1. **בחר את הטאב "Packet Extraction"**
   - Select the "Packet Extraction" tab
   - **לא** את "Vector Building" !

2. **לחץ על "Select MAT File"**
   - Click on "Select MAT File"

3. **בחר את קובץ הוקטור**:
   - עבור לתיקיית data/
   - בחר קובץ וקטור (למשל: vector_XXXXX.mat)
   - לחץ "Open"

4. **לחץ "Open Spectrogram and Cut Packet"**
   - Click "Open Spectrogram and Cut Packet"

5. **🎉 חלון ספקטרוגרמה יפתח!**
   - הוקטור יוצג עם כל הפקטות
   - זיהוי אוטומטי של גבולות הפקטות
   - רזולוציה של מיקרושניות

---

## 🔧 מה תוקן / What Was Fixed

### הבעיה המקורית:
- וקטורים נשמרו בלי `pre_samples`
- packet extractor לא יכול היה לטעון אותם

### התיקון:
```python
# קודם:
sio.savemat(output_path, {'Y': vector})

# אחרי התיקון:
sio.savemat(output_path, {
    'Y': vector,
    'pre_samples': 0  # תואם לפקטות
})
```

---

## 🎯 תכונות זמינות / Available Features

### עכשיו אפשר:
1. **ליצור וקטורים** עם מספר פקטות
2. **לפתוח וקטורים** ב-packet extractor
3. **לראות ספקטרוגרמה** מפורטת
4. **לזהות פקטות** אוטומטית
5. **לחתוך פקטות** מהוקטור
6. **לשמור פקטות** נוספות

### Advanced Features:
- **Microsecond precision** in packet timing
- **Automatic bounds detection** for packets
- **High-resolution spectrograms** (1μs time resolution)
- **Interactive cutting** with g/r keys
- **Real-time visualization** of packet content

---

## 🧪 בדיקה מהירה / Quick Test

### צור וקטור בדיקה:
```bash
source venv/bin/activate
python -c "
import numpy as np
from utils import save_vector
import os

# צור וקטור דוגמה
vector = np.exp(2j * np.pi * 5e6 * np.arange(140000) / 56e6)
os.makedirs('data', exist_ok=True)
save_vector(vector, 'data/test_vector.mat')
print('✅ Test vector created: data/test_vector.mat')
"
```

### פתח בGUI:
1. הפעל: `python unified_gui.py`
2. Packet Extraction tab
3. Select MAT File → data/test_vector.mat
4. Open Spectrogram and Cut Packet
5. 🎉 Success!

---

## 📊 מצב הקבצים / File Status

### כל הקבצים תקינים:
- ✅ `packet_1_5MHz.mat` - Y=True, pre_samples=True
- ✅ `packet_2_chirp.mat` - Y=True, pre_samples=True
- ✅ `packet_3_bpsk.mat` - Y=True, pre_samples=True
- ✅ `packet_4_noise.mat` - Y=True, pre_samples=True
- ✅ `packet_5_multitone.mat` - Y=True, pre_samples=True
- ✅ `test_vector_for_gui.mat` - Y=True, pre_samples=True

### כל הקבצים ניתנים לפתיחה בpacket extractor!

---

## ❓ פתרון בעיות / Troubleshooting

### אם זה עדיין לא עובד:

#### בדיקה 1: וודא שהטאב נכון
- **צריך**: Packet Extraction tab
- **לא**: Vector Building tab

#### בדיקה 2: וודא שהקובץ קיים
```bash
ls -la data/*.mat
```

#### בדיקה 3: בדוק שהקובץ תקין
```python
import scipy.io as sio
data = sio.loadmat('data/your_vector.mat')
print('Keys:', list(data.keys()))
```

#### בדיקה 4: רענן את רשימת הקבצים
- לחץ "Refresh Packet List" בGUI

---

## 🚀 הצלחה! / Success!

**המערכת עובדת במלואה!**  
**The system is fully functional!**

- יצירת וקטורים ✅
- פתיחת וקטורים ✅
- זיהוי פקטות ✅
- חיתוך פקטות ✅
- שמירה אוטומטית ✅

**המערכת מוכנה לשימוש מלא! 🎉**  
**The system is ready for full use! 🎉**

---

*מעודכן ב-* `$(date)` *- כל הבעיות נפתרו*  
*Updated on* `$(date)` *- All issues resolved*