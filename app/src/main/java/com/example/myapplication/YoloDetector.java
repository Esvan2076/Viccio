package com.example.myapplication;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

/**
 * YOLOv8 detector using TensorFlow Lite.
 *
 * Mirrors the Python script logic:
 *   model = YOLO("best_float32.tflite")
 *   results = model(frame, imgsz=320, conf=0.5, verbose=False)
 *
 * The model expects input of shape [1, 320, 320, 3] with float32 pixel values
 * normalized to [0, 1].
 *
 * YOLOv8 TFLite output shape is [1, 4+num_classes, 8400] where:
 *   - First 4 values per detection: cx, cy, w, h (center x, center y, width, height)
 *   - Remaining values: class probabilities
 *   - 8400 = total number of anchor boxes across all scales
 */
public class YoloDetector {

    private static final String TAG = "YoloDetector";

    // Model configuration — matches the Python script: imgsz=320, conf=0.5
    private static final String MODEL_FILE = "best_float32_500.tflite";
    private static final int INPUT_SIZE = 320;
    private static final float CONFIDENCE_THRESHOLD = 0.5f;
    private static final float IOU_THRESHOLD = 0.45f; // NMS threshold (YOLOv8 default)

    private Interpreter interpreter;
    private int numClasses;
    private int numDetections; // 8400 for YOLOv8
    private String[] classNames;

    // Reusable buffers to avoid per-frame allocations
    private ByteBuffer inputBuffer;

    /**
     * Initialize the detector by loading the TFLite model from assets.
     *
     * @param context Application context
     * @param labels  Array of class names corresponding to the model's output classes
     * @throws IOException if the model file cannot be loaded
     */
    public YoloDetector(Context context, String[] labels) throws IOException {
        this.classNames = labels;

        // Load model
        MappedByteBuffer modelBuffer = loadModelFile(context);

        // Configure interpreter
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(4); // Use 4 CPU threads for inference
        interpreter = new Interpreter(modelBuffer, options);

        // Inspect model input/output shapes
        int[] inputShape = interpreter.getInputTensor(0).shape(); // [1, 320, 320, 3]
        int[] outputShape = interpreter.getOutputTensor(0).shape(); // [1, 4+num_classes, 8400]

        Log.d(TAG, "Model input shape: [" + inputShape[0] + ", " + inputShape[1] + ", "
                + inputShape[2] + ", " + inputShape[3] + "]");
        Log.d(TAG, "Model output shape: [" + outputShape[0] + ", " + outputShape[1] + ", "
                + outputShape[2] + "]");

        // YOLOv8 output: [1, 4+num_classes, 8400]
        int outputDim1 = outputShape[1]; // 4 + num_classes
        numDetections = outputShape[2];  // 8400
        numClasses = outputDim1 - 4;

        Log.d(TAG, "Number of classes: " + numClasses);
        Log.d(TAG, "Number of detections: " + numDetections);

        // Prepare input buffer: 1 * 320 * 320 * 3 * 4 bytes (float32)
        inputBuffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4);
        inputBuffer.order(ByteOrder.nativeOrder());

        Log.d(TAG, "YoloDetector initialized successfully");
    }

    /**
     * Run inference on a bitmap frame.
     * This is the equivalent of: results = model(frame, imgsz=320, conf=0.5)
     *
     * @param bitmap The camera frame to analyze
     * @return List of DetectionResult objects that passed the confidence threshold
     */
    public List<DetectionResult> detect(Bitmap bitmap) {
        // 1. Preprocess: Resize to 320x320 and fill input buffer
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        fillInputBuffer(resized);
        if (resized != bitmap) {
            resized.recycle();
        }

        // 2. Prepare output buffer: [1, 4+numClasses, 8400]
        float[][][] output = new float[1][4 + numClasses][numDetections];

        // 3. Run inference
        interpreter.run(inputBuffer, output);

        // 4. Post-process: extract detections above confidence threshold
        List<DetectionResult> rawDetections = postProcess(output[0]);

        // 5. Apply Non-Maximum Suppression (NMS)
        List<DetectionResult> nmsResults = applyNMS(rawDetections, IOU_THRESHOLD);

        return nmsResults;
    }

    /**
     * Fill the input ByteBuffer with normalized pixel values from the bitmap.
     * Normalizes pixel values from [0, 255] to [0.0, 1.0] as expected by YOLOv8.
     */
    private void fillInputBuffer(Bitmap bitmap) {
        inputBuffer.rewind();

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            // Extract RGB channels and normalize to [0, 1]
            inputBuffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
            inputBuffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
            inputBuffer.putFloat((pixel & 0xFF) / 255.0f);          // B
        }
    }

    /**
     * Post-process YOLOv8 output.
     * Output shape is [4+num_classes, 8400] (transposed compared to PyTorch).
     *
     * For each of the 8400 detections:
     *   - output[0..3][i] = cx, cy, w, h (in input pixel coordinates, 0-320)
     *   - output[4..4+num_classes-1][i] = class probabilities
     *
     * Apply confidence threshold (0.5) as specified in the Python script.
     */
    private List<DetectionResult> postProcess(float[][] output) {
        List<DetectionResult> results = new ArrayList<>();

        for (int i = 0; i < numDetections; i++) {
            // Find the class with the highest confidence
            float maxConf = 0f;
            int maxClassIdx = 0;

            for (int c = 0; c < numClasses; c++) {
                float conf = output[4 + c][i];
                if (conf > maxConf) {
                    maxConf = conf;
                    maxClassIdx = c;
                }
            }

            // Apply confidence threshold (conf=0.5 from the Python script)
            if (maxConf < CONFIDENCE_THRESHOLD) {
                continue;
            }

            // Extract bounding box (center format) and convert to corner format
            float cx = output[0][i];
            float cy = output[1][i];
            float w = output[2][i];
            float h = output[3][i];

            // Convert from center format to corner format, normalized to [0, 1]
            float left = (cx - w / 2f) / INPUT_SIZE;
            float top = (cy - h / 2f) / INPUT_SIZE;
            float right = (cx + w / 2f) / INPUT_SIZE;
            float bottom = (cy + h / 2f) / INPUT_SIZE;

            // Clamp to [0, 1]
            left = Math.max(0f, Math.min(1f, left));
            top = Math.max(0f, Math.min(1f, top));
            right = Math.max(0f, Math.min(1f, right));
            bottom = Math.max(0f, Math.min(1f, bottom));

            // Get class name
            String className;
            if (classNames != null && maxClassIdx < classNames.length) {
                className = classNames[maxClassIdx];
            } else {
                className = "class_" + maxClassIdx;
            }

            results.add(new DetectionResult(className, maxConf, new RectF(left, top, right, bottom)));
        }

        return results;
    }

    /**
     * Apply Non-Maximum Suppression to remove overlapping detections.
     * Uses greedy NMS with IoU threshold.
     */
    private List<DetectionResult> applyNMS(List<DetectionResult> detections, float iouThreshold) {
        if (detections.isEmpty()) {
            return detections;
        }

        // Sort by confidence (highest first)
        detections.sort((a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));

        List<DetectionResult> selected = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];

        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;

            selected.add(detections.get(i));

            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;

                float iou = computeIoU(
                        detections.get(i).getBoundingBox(),
                        detections.get(j).getBoundingBox()
                );

                if (iou > iouThreshold) {
                    suppressed[j] = true;
                }
            }
        }

        return selected;
    }

    /**
     * Compute Intersection over Union (IoU) between two bounding boxes.
     */
    private float computeIoU(RectF boxA, RectF boxB) {
        float intersectionLeft = Math.max(boxA.left, boxB.left);
        float intersectionTop = Math.max(boxA.top, boxB.top);
        float intersectionRight = Math.min(boxA.right, boxB.right);
        float intersectionBottom = Math.min(boxA.bottom, boxB.bottom);

        float intersectionArea = Math.max(0, intersectionRight - intersectionLeft) *
                Math.max(0, intersectionBottom - intersectionTop);

        float boxAArea = (boxA.right - boxA.left) * (boxA.bottom - boxA.top);
        float boxBArea = (boxB.right - boxB.left) * (boxB.bottom - boxB.top);

        float unionArea = boxAArea + boxBArea - intersectionArea;

        if (unionArea <= 0) return 0f;

        return intersectionArea / unionArea;
    }

    /**
     * Load the TFLite model file from assets as a memory-mapped buffer.
     * This is more efficient than reading the entire file into memory.
     */
    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        inputStream.close();
        fileDescriptor.close();
        return buffer;
    }

    /**
     * Release resources held by the interpreter.
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
