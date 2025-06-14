# מחולל וקטורים מפקטות

אפליקציה ליצירת וקטורים מורכבים מפקטות שונות.

## תכונות עיקריות
- טעינת פקטות מקובצי .mat
- תמיכה ב-6 סוגי פקטות שונים
- התאמת Sample-Rate אוטומטית
- יצירת וקטור מורכב בקצב דגימה סופי של 56MHz
- הגדרת היסט התחלתי בזמן לכל פקטה
- הצגת ספקטוגרמה
- שמירת התוצאה כקובץ .mat או .wv

## התקנה
```bash
pip install -r requirements.txt
```

## שימוש
הרץ את הקובץ main.py:
```bash
python main.py
```

בסיומה של בניית הוקטור ניתן לבחור האם לשמור אותו בפורמט MAT או WV באמצעות שני כפתורים נפרדים באפליקציה.
קצב הדגימה הסופי קבוע על 56MHz כברירת מחדל אך ניתן לשנותו על פי הצורך.

## מבנה הנתונים
- הפקטות צריכות להיות בקובצי .mat
- המידע נמצא במשתנה Y בקובץ
- הנתונים הם קומפלקסיים

## בדיקות
לאחר התקנת התלויות ניתן להריץ את מערך הבדיקות באמצעות:
```bash
pytest
```
