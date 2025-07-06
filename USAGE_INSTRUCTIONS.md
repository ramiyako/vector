# הוראות הפעלה - מערכת חילוץ פקטות ויצירת וקטורים
# Usage Instructions - Packet Extraction and Vector Generation System

## התחלה מהירה / Quick Start

### 1. הפעלת המערכת / System Startup

```bash
# הפעלת הסביבה הווירטואלית / Activate virtual environment
source venv/bin/activate

# הפעלת הממשק הגרפי / Launch GUI
python unified_gui.py
```

### 2. חילוץ פקטה מקובץ MAT / Extract Packet from MAT File

1. **בחירת קובץ** / **File Selection**:
   - לחץ על "Select MAT File"
   - בחר קובץ .mat עם האות

2. **הצגת ספקטרוגרמה** / **Show Spectrogram**:
   - לחץ על "Open Spectrogram and Cut Packet"
   - **חלון אינטראקטיבי ייפתח אוטומטית**

3. **חיתוך הפקטה** / **Cut the Packet**:
   - לחץ 'g' לבחירת הקו הירוק (התחלה)
   - לחץ 'r' לבחירת הקו האדום (סוף)
   - גרור את הקורסורים למיקום הרצוי
   - לחץ Enter לסיום

4. **שמירה אוטומטית** / **Automatic Saving**:
   - **הפקטה תישמר אוטומטית בסגירת החלון**
   - גם אם לא שיניתם את מיקום הקורסורים!

### 3. יצירת וקטור / Vector Creation

1. **עבור לטאב "Vector Building"**
2. **הגדר פרמטרים כלליים**:
   - אורך וקטור: **2.5** מילישניות
   - מספר פקטות: בחר כמה פקטות רוצה

3. **הגדר כל פקטה**:
   - בחר קובץ פקטה
   - הגדר תדר הסטה (MHz)
   - הגדר תקופה חזרה (ms)
   - הגדר זמן התחלה (ms)

4. **יצירת הוקטור**:
   - לחץ "Create MAT Vector" או "Create WV Vector"
   - **ספקטרוגרמה תוצג אוטומטית**

### 4. בדיקת דיוק הזמן / Verify Timing Precision

הוקטור שנוצר יציג:
- **מרווחי זמן מדויקים של 1 מיקרושניה**
- סמנים המציינים את מיקום הפקטות
- מידע מפורט על הזמנים

---

## תכונות מתקדמות / Advanced Features

### זיהוי אוטומטי של גבולות / Automatic Bounds Detection
- המערכת מזהה אוטומטית את תחילת וסוף הפקטה
- אפשר להשתמש בזיהוי האוטומטי או לכוונן ידנית
- דיוק גבוה בכל המקרים

### רזולוציה גבוהה / High Resolution
- **רזולוציית זמן: 1 מיקרושניה**
- רזולוציית תדר: עד 27 kHz
- תצוגה אינטראקטיבית וברורה

### שמירה חכמה / Smart Saving
- שמירה אוטומטית בסגירת חלונות
- וידוא תקינות הנתונים
- תמיכה בפורמטים MAT ו-WV

---

## פתרון בעיות / Troubleshooting

### חלון לא נפתח / Window Not Opening
```bash
# וודא שה-X11 פועל (במערכות Linux)
export DISPLAY=:0

# או השתמש בבדיקה ללא GUI
python test_complete_workflow.py
```

### שגיאות טעינה / Loading Errors
- וודא שקובץ ה-MAT תקין
- בדוק שהמשתנה נקרא 'Y' או שהוא המשתנה היחיד

### בעיות דיוק / Precision Issues
- השתמש בתדר דגימה 56 MHz
- וודא שהפקטות קצרות יחסית (עד כמה ms)

---

## דוגמאות שימוש / Usage Examples

### דוגמה 1: פקטה יחידה
```
1. טען packet_1_5MHz.mat
2. חתוך את הפקטה (או השאר אוטומטי)
3. הפקטה תישמר ב-data/
```

### דוגמה 2: וקטור משתי פקטות
```
Vector Length: 2.5 ms

Packet 1:
- File: packet_1_5MHz.mat
- Frequency Shift: 0 MHz
- Period: 1.0 ms
- Start Time: 0.2 ms

Packet 2:
- File: packet_2_chirp.mat  
- Frequency Shift: 10 MHz
- Period: 1.2 ms
- Start Time: 0.6 ms
```

### תוצאה: וקטור באורך 2.5ms עם פקטות במיקומים מדויקים

---

## בדיקת המערכת / System Testing

### בדיקה מהירה / Quick Test
```bash
source venv/bin/activate
python test_complete_workflow.py
```

### בדיקה מלאה / Full Test
```bash
# צור פקטות דוגמה
python create_test_packets.py

# הרץ בדיקה מלאה
python test_complete_workflow.py

# הפעל GUI
python unified_gui.py
```

---

## טיפים לשימוש אופטימלי / Optimization Tips

### 1. **דיוק זמן מקסימלי**:
   - השתמש בתדר דגימה 56 MHz
   - הגדר זמנים במדיוק של מיקרושניות
   - בדוק תמיד את התוצאות

### 2. **איכות ספקטרוגרמה**:
   - השתמש ברזולוציית זמן 1 μs
   - וודא שהחלונות נפתחים במסך גדול
   - בדוק את כל הפרמטרים

### 3. **ניהול קבצים**:
   - שמור פקטות עם שמות ברורים
   - בדוק גדלי קבצים
   - נקה קבצים זמניים

### 4. **ביצועים**:
   - סגור יישומים מיותרים
   - השתמש ב-SSD אם אפשר
   - וודא זיכרון פנוי מספיק

---

## תמיכה ועזרה / Support and Help

- **קרא את הדוח המלא**: WORKFLOW_VERIFICATION_REPORT.md
- **בדוק דוגמאות**: תיקיית tests/
- **הרץ בדיקות**: python test_complete_workflow.py

המערכת נבדקה במלואה ועוברת את כל הבדיקות! ✅

---

**לשימוש קל ויעיל** 🚀