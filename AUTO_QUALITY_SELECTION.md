# Automatic Quality Selection Feature

## תיאור בעברית

הוספתי תכונה חדשה של בחירת איכות אוטומטית לאפליקציית חילוץ הפקטים. התכונה מחליטה אוטומטית על רמת האיכות על בסיס גודל הקובץ והזמן הצפוי לניתוח.

### יכולות חדשות:
1. **בחירת איכות אוטומטית** - האפליקציה מחליטה לבד על רמת האיכות המתאימה
2. **החלטה מבוססת גודל קובץ** - קבצים גדולים יותר יקבלו איכות נמוכה יותר למהירות
3. **החלטה מבוססת זמן ניתוח** - ניתוחים ארוכים יותר יקבלו איכות נמוכה יותר
4. **אפשרות לעקוף** - המשתמש יכול לשנות את הבחירה האוטומטית

### לוגיקת ההחלטה:
- **קבצים קטנים** (< 50MB, < 10s ניתוח) → איכות גבוהה
- **קבצים בינוניים** (50-200MB, 10-30s ניתוח) → איכות מאוזנת
- **קבצים גדולים** (> 200MB, > 30s ניתוח) → איכות מהירה

---

## English Description

I've added a new automatic quality selection feature to the packet extractor application. The feature automatically decides on the quality level based on file size and estimated analysis time.

### New Features:
1. **Automatic Quality Selection** - The application automatically decides on the appropriate quality level
2. **File Size-Based Decision** - Larger files get lower quality for faster processing
3. **Analysis Time-Based Decision** - Longer analysis times get lower quality
4. **Override Capability** - Users can manually change the automatic selection

### Decision Logic:
- **Small files** (< 50MB, < 10s analysis) → High Quality
- **Medium files** (50-200MB, 10-30s analysis) → Balanced
- **Large files** (> 200MB, > 30s analysis) → Fast

## Technical Implementation

### Code Changes

#### 1. New Variables Added:
```python
self.auto_quality_enabled = tk.BooleanVar(value=True)  # Enable automatic quality decision
```

#### 2. New UI Elements:
- **Auto Quality Selection Checkbox** - Enables/disables automatic quality selection
- **Performance Label** - Shows auto-selected quality and reason
- **Enhanced Notifications** - Detailed information about automatic selection

#### 3. New Function: `auto_determine_quality()`
```python
def auto_determine_quality(self, file_size_mb, signal_length):
    """Automatically determine quality based on file size and estimated analysis time"""
    
    # Estimate analysis time based on signal length
    estimated_time_fast = signal_length / 10_000_000  # Fast mode: ~10M samples per second
    estimated_time_balanced = signal_length / 5_000_000  # Balanced: ~5M samples per second
    estimated_time_high = signal_length / 2_000_000  # High quality: ~2M samples per second
    
    # Decision logic based on file size and analysis time
    if file_size_mb > 200 or estimated_time_high > 30:
        recommended_quality = "Fast"
        reason = f"Large file ({file_size_mb:.1f}MB) or long analysis time ({estimated_time_high:.1f}s)"
    elif file_size_mb > 50 or estimated_time_balanced > 10:
        recommended_quality = "Balanced"
        reason = f"Medium file ({file_size_mb:.1f}MB) or moderate analysis time ({estimated_time_balanced:.1f}s)"
    else:
        recommended_quality = "High Quality"
        reason = f"Small file ({file_size_mb:.1f}MB) allowing high quality analysis"
    
    return recommended_quality, reason, time_estimates
```

#### 4. Enhanced `load_file()` Function:
- Automatically determines quality when file is loaded
- Applies the recommended quality settings
- Shows detailed notification with reasoning
- Provides estimated analysis times for all quality levels

### Quality Decision Matrix

| File Size | Analysis Time (High Quality) | Selected Quality | Reason |
|-----------|------------------------------|------------------|--------|
| < 50MB | < 10s | High Quality | Small file, fast analysis |
| 50-200MB | 10-30s | Balanced | Medium file, moderate analysis |
| > 200MB | > 30s | Fast | Large file, long analysis |

### User Experience

#### When File is Loaded:
1. **File size is calculated** automatically
2. **Signal length is determined** from the loaded data
3. **Analysis time is estimated** for each quality level
4. **Quality is automatically selected** based on the decision matrix
5. **User is notified** with detailed information about the selection
6. **Quality settings are applied** immediately

#### Notification Message:
```
🤖 Auto Quality Selection:

Selected: Balanced
Reason: Medium file (75.2MB) or moderate analysis time (15.3s)

Estimated analysis times:
• Fast: 5.6s
• Balanced: 11.2s
• High Quality: 28.0s

You can change the quality setting manually if needed.
```

### User Controls

#### Auto Quality Selection Checkbox:
- **Enabled (default)**: Automatic quality selection is active
- **Disabled**: Manual quality selection only

#### Quality Override:
- Users can manually change the quality after automatic selection
- The performance label shows when auto-selection is active
- Manual changes override the automatic selection

### Benefits

1. **Optimal Performance**: Files get the best quality that can be processed efficiently
2. **User-Friendly**: No need to manually decide on quality for each file
3. **Time-Saving**: Automatic decision reduces user interaction time
4. **Informed Decision**: Clear reasoning provided for each selection
5. **Flexible**: Users can still override if needed

### Usage Instructions

1. **Enable Auto Quality**: Check the "Auto Quality Selection" checkbox (enabled by default)
2. **Load File**: Select a MAT file using the "Select MAT File" button
3. **View Selection**: The application will automatically select and apply quality settings
4. **Review Information**: Check the notification message for detailed reasoning
5. **Override if Needed**: Manually change quality settings if desired
6. **Proceed**: Continue with packet extraction using the selected quality

### Performance Expectations

#### Small Files (< 50MB):
- **Selected Quality**: High Quality
- **Benefits**: Maximum precision, detailed analysis
- **Trade-offs**: Slightly longer processing time

#### Medium Files (50-200MB):
- **Selected Quality**: Balanced
- **Benefits**: Good compromise between speed and quality
- **Trade-offs**: Moderate processing time and precision

#### Large Files (> 200MB):
- **Selected Quality**: Fast
- **Benefits**: Quick processing, reduced memory usage
- **Trade-offs**: Lower precision, faster results

### Testing

#### Test Script: `test_auto_quality.py`
- Creates test files of different sizes
- Demonstrates automatic quality selection
- Verifies correct quality selection for each file size
- Shows estimated analysis times

#### Usage:
```bash
python test_auto_quality.py
```

### Integration with Existing Features

The automatic quality selection integrates seamlessly with:
- **Quality Presets**: Auto-selection applies the appropriate preset
- **Manual Override**: Users can change quality after auto-selection
- **Quality Testing**: Test button works with auto-selected quality
- **Advanced Controls**: All manual controls remain available

### Future Enhancements

Potential improvements:
1. **Machine Learning**: Learn from user overrides to improve decisions
2. **File Type Detection**: Different logic for different signal types
3. **Memory Usage Prediction**: Factor in available system memory
4. **Processing Speed History**: Learn from actual processing times
5. **Custom Decision Rules**: Allow users to define custom decision logic

## Conclusion

The automatic quality selection feature successfully implements the requested functionality:
- ✅ Automatic quality decision based on file size and analysis time
- ✅ User can override the automatic selection
- ✅ Clear reasoning provided for each decision
- ✅ Seamless integration with existing workflow
- ✅ Maintains all existing functionality

The feature makes the application more user-friendly by automatically selecting optimal quality settings while still allowing full user control when needed.