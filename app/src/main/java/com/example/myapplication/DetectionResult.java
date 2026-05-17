package com.example.myapplication;

import android.graphics.RectF;

/**
 * Represents a single object detection result from the YOLOv8 model.
 * Mirrors the detection output from the Python script: class name, confidence, and bounding box.
 */
public class DetectionResult {

    private final String className;
    private final float confidence;
    private final RectF boundingBox; // Normalized coordinates [0.0, 1.0]

    /**
     * @param className  The detected class name (e.g., "persona", "coche", etc.)
     * @param confidence The confidence score [0.0, 1.0]
     * @param boundingBox The bounding box in normalized coordinates relative to the input image
     */
    public DetectionResult(String className, float confidence, RectF boundingBox) {
        this.className = className;
        this.confidence = confidence;
        this.boundingBox = boundingBox;
    }

    public String getClassName() {
        return className;
    }

    public float getConfidence() {
        return confidence;
    }

    /**
     * Returns the confidence as a percentage (0-100).
     */
    public float getConfidencePercent() {
        return confidence * 100f;
    }

    public RectF getBoundingBox() {
        return boundingBox;
    }

    @Override
    public String toString() {
        return String.format("%s: %.1f%%", className, getConfidencePercent());
    }
}
