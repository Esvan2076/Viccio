package com.example.myapplication;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * Custom view that draws bounding boxes and labels on top of the camera preview.
 * This replicates the annotated_frame = results[0].plot() from the Python script,
 * drawing detection results directly over the live camera feed.
 */
public class OverlayView extends View {

    // Predefined colors for different classes (cycling through them)
    private static final int[] COLORS = {
            Color.rgb(255, 56, 56),    // Red
            Color.rgb(255, 157, 56),   // Orange
            Color.rgb(255, 255, 56),   // Yellow
            Color.rgb(56, 255, 56),    // Green
            Color.rgb(56, 255, 255),   // Cyan
            Color.rgb(56, 56, 255),    // Blue
            Color.rgb(255, 56, 255),   // Magenta
            Color.rgb(255, 112, 112),  // Light Red
            Color.rgb(255, 194, 112),  // Light Orange
            Color.rgb(112, 255, 112),  // Light Green
    };

    private final Paint boxPaint;
    private final Paint textPaint;
    private final Paint textBackgroundPaint;

    private List<DetectionResult> results = new ArrayList<>();

    public OverlayView(Context context) {
        this(context, null);
    }

    public OverlayView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public OverlayView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);

        // Paint for bounding boxes
        boxPaint = new Paint();
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);
        boxPaint.setAntiAlias(true);

        // Paint for label text
        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40f);
        textPaint.setAntiAlias(true);
        textPaint.setFakeBoldText(true);

        // Paint for text background
        textBackgroundPaint = new Paint();
        textBackgroundPaint.setStyle(Paint.Style.FILL);
    }

    /**
     * Update the detection results to be drawn on the next frame.
     * Thread-safe: can be called from the analysis thread.
     *
     * @param detectionResults List of DetectionResult objects to visualize
     */
    public void setResults(List<DetectionResult> detectionResults) {
        this.results = new ArrayList<>(detectionResults);
        postInvalidate(); // Request redraw on UI thread
    }

    /**
     * Clear all drawn detections.
     */
    public void clear() {
        this.results = new ArrayList<>();
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        int viewWidth = getWidth();
        int viewHeight = getHeight();

        for (int i = 0; i < results.size(); i++) {
            DetectionResult result = results.get(i);
            int color = COLORS[i % COLORS.length];

            // Set color for this detection
            boxPaint.setColor(color);
            textBackgroundPaint.setColor(color);

            // Convert normalized coordinates to view coordinates
            RectF normalizedBox = result.getBoundingBox();
            RectF scaledBox = new RectF(
                    normalizedBox.left * viewWidth,
                    normalizedBox.top * viewHeight,
                    normalizedBox.right * viewWidth,
                    normalizedBox.bottom * viewHeight
            );

            // Draw bounding box
            canvas.drawRect(scaledBox, boxPaint);

            // Prepare label text: "className 85.3%"
            String label = String.format("%s %.1f%%", result.getClassName(), result.getConfidencePercent());

            // Measure text for background
            float textWidth = textPaint.measureText(label);
            float textHeight = textPaint.getTextSize();
            float padding = 8f;

            // Draw label background
            float labelTop = Math.max(scaledBox.top - textHeight - padding * 2, 0);
            canvas.drawRect(
                    scaledBox.left,
                    labelTop,
                    scaledBox.left + textWidth + padding * 2,
                    labelTop + textHeight + padding * 2,
                    textBackgroundPaint
            );

            // Draw label text
            canvas.drawText(
                    label,
                    scaledBox.left + padding,
                    labelTop + textHeight + padding,
                    textPaint
            );
        }
    }
}
