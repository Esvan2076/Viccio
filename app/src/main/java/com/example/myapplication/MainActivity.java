package com.example.myapplication;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.text.TextRecognition;
import com.google.mlkit.vision.text.TextRecognizer;
import com.google.mlkit.vision.text.latin.TextRecognizerOptions;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "BichioMain";

    // ── CONFIGURACIÓN ESP32-CAM ──────────────────────────────────────────────
    private static final String STREAM_URL = "http://10.50.172.39:81/stream";
    private float cameraRotationAngle = 0.0f; // Ajusta el ángulo si tu cámara está de lado
    private boolean isStreaming = false;

    // ── UI ───────────────────────────────────────────────────────────────────
    private ImageView streamImageView;
    private OverlayView overlayView;
    private TextView    tvDetected;
    private TextView    tvInference;

    // ── Ejecutores Separados (Evita retrasos en la lectura de red) ────────────
    private ExecutorService streamExecutor;
    private ExecutorService aiExecutor;

    // ── OCR (ML Kit) ─────────────────────────────────────────────────────────
    private TextRecognizer textRecognizer;

    // ── YOLO (TFLite) ────────────────────────────────────────────────────────
    private YoloDetector yoloDetector;
    private String[]     classNames;
    private boolean      yoloReady = false;

    // ── TTS ──────────────────────────────────────────────────────────────────
    private TextToSpeech tts;
    private boolean      isTtsReady = false;
    private boolean      isSpeaking = false;

    // ── STT ──────────────────────────────────────────────────────────────────
    private SpeechRecognizer speechRecognizer;
    private Intent           speechRecognizerIntent;
    private boolean          isListening = false;

    // ── Estado general ───────────────────────────────────────────────────────
    private boolean isSystemActive  = false;
    private boolean isReading       = false;
    private boolean isRestroomMode  = false;

    // ================= OCR STABILITY =================
    private int    stableCount   = 0;
    private static final int    STABLE_THRESHOLD = 2;
    private String lastScene     = "";
    private long   lastSceneTime = 0;
    private static final long   SCENE_COOLDOWN = 1500;

    // Anti-rebote para anuncio YOLO
    private String lastYoloScene     = "";
    private long   lastYoloSceneTime = 0;
    private static final long YOLO_COOLDOWN = 3000;

    // ================= WAKE =================
    private String lastWake     = "";
    private long   lastWakeTime = 0;
    private static final long  WAKE_COOLDOWN   = 1500;

    // ================= FPS CONTROL =================
    private long lastFrameTime = 0;
    private static final long FRAME_INTERVAL = 150;

    private final Handler handler = new Handler(Looper.getMainLooper());

    // ════════════════════════════════════════════════════════════════════════
    // onCreate
    // ════════════════════════════════════════════════════════════════════════

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        streamImageView = findViewById(R.id.streamImageView);
        overlayView = findViewById(R.id.overlayView);
        tvDetected  = findViewById(R.id.tvDetected);
        tvInference = findViewById(R.id.tvInferenceTime);

        // Hilos de ejecución separados para fluidez total
        streamExecutor = Executors.newSingleThreadExecutor();
        aiExecutor     = Executors.newSingleThreadExecutor();

        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

        setupTTS();
        setupSpeechRecognizer();

        classNames = loadLabels();
        loadYoloModel();

        requestPermissionsIfNeeded();
    }

    // ════════════════════════════════════════════════════════════════════════
    // TTS (Motor de Voz)
    // ════════════════════════════════════════════════════════════════════════

    private void setupTTS() {
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                int result = tts.setLanguage(new Locale("es", "MX"));
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    tts.setLanguage(Locale.getDefault());
                }
                isTtsReady = true;
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override public void onStart(String utteranceId) { isSpeaking = true; }
                    @Override public void onDone(String utteranceId)  { isSpeaking = false; }
                    @Override public void onError(String utteranceId) { isSpeaking = false; }
                });
                handler.postDelayed(() -> speak("Listo"), 800);
            } else {
                Log.e(TAG, "Error crítico al inicializar TextToSpeech.");
            }
        });
    }

    private void speak(String text) {
        if (tts == null || !isTtsReady) return;
        try {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "BichioUtteranceId");
        } catch (Exception e) {
            Log.e(TAG, "Error en reproducción TTS", e);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // STT (Control de voz)
    // ════════════════════════════════════════════════════════════════════════

    private void setupSpeechRecognizer() {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);
        speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "es-MX");

        speechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override public void onReadyForSpeech(Bundle p)  { isListening = true; }
            @Override public void onBeginningOfSpeech()        {}
            @Override public void onRmsChanged(float r)        {}
            @Override public void onBufferReceived(byte[] b)   {}
            @Override public void onEvent(int t, Bundle p)     {}
            @Override public void onEndOfSpeech() { isListening = false; restartListening(); }
            @Override public void onError(int error) { isListening = false; restartListening(); }
            @Override public void onResults(Bundle results) { handleVoice(results); restartListening(); }
            @Override public void onPartialResults(Bundle partialResults) { handleVoice(partialResults); }
        });
    }

    private void handleVoice(Bundle results) {
        if (isSpeaking && tts.isSpeaking()) return;
        ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        if (matches == null) return;
        long now = System.currentTimeMillis();

        for (String m : matches) {
            String text = normalize(m);

            if (text.contains("INICIAR") || text.contains("START")) {
                if (isSystemActive) return;
                if (lastWake.equals("INICIAR") && (now - lastWakeTime) < WAKE_COOLDOWN) return;
                lastWake = "INICIAR"; lastWakeTime = now; isSystemActive = true;
                speak("Sistema encendido"); return;
            }
            if (text.contains("LEER") || text.contains("TEXTO") || text.contains("SIGN")) {
                if (!isSystemActive || isReading) return;
                isReading = true; isRestroomMode = false; lastScene = ""; stableCount = 0;
                runOnUiThread(() -> {
                    if (overlayView != null) overlayView.setResults(new ArrayList<>());
                    if (tvDetected != null) tvDetected.setText("");
                });
                speak("Lectura de señales activada"); return;
            }
            if (text.contains("BANO") || text.contains("BANOS") || text.contains("RESTROOM")) {
                if (!isSystemActive || isRestroomMode) return;
                if (!yoloReady) { speak("Modelo no listo, espera un momento"); return; }
                isRestroomMode = true; isReading = false; lastScene = ""; stableCount = 0;
                runOnUiThread(() -> { if (tvDetected != null) tvDetected.setText(""); });
                speak("Búsqueda de baños activada"); return;
            }
            if (text.contains("APAGAR") || text.contains("DETENER") || text.contains("STOP")) {
                if (!isSystemActive) return;
                isSystemActive = false; isReading = false; isRestroomMode = false;
                if (tts.isSpeaking()) tts.stop();
                runOnUiThread(() -> { if (overlayView != null) overlayView.setResults(new ArrayList<>()); });
                speak("Sistema apagado"); return;
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Lector de Stream HTTP (Reemplaza por completo CameraX)
    // ════════════════════════════════════════════════════════════════════════

    private void startMjpegStream() {
        isStreaming = true;
        streamExecutor.execute(() -> {
            HttpURLConnection connection = null;
            BufferedInputStream bis = null;
            try {
                URL url = new URL(STREAM_URL);
                connection = (HttpURLConnection) url.openConnection();
                connection.setConnectTimeout(5000);
                connection.setReadTimeout(5000);

                bis = new BufferedInputStream(connection.getInputStream());
                ByteArrayOutputStream jpegBuffer = new ByteArrayOutputStream();
                int prevByte = 0, currentByte;
                boolean isCaptureStarted = false;

                Log.d(TAG, "Conectado al stream de la ESP32.");

                while (isStreaming && (currentByte = bis.read()) != -1) {
                    if (isCaptureStarted) {
                        jpegBuffer.write(currentByte);
                        if (prevByte == 0xFF && currentByte == 0xD9) {
                            byte[] rawImage = jpegBuffer.toByteArray();
                            Bitmap rawBitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length);

                            if (rawBitmap != null) {
                                // 1. Corregir ángulo de rotación
                                final Bitmap rotatedBitmap = rotateBitmap(rawBitmap, cameraRotationAngle);

                                // 2. Dibujar el video inmediatamente en pantalla
                                runOnUiThread(() -> streamImageView.setImageBitmap(rotatedBitmap));

                                // 3. Despachar a la IA respetando el intervalo de FPS asignado
                                long now = System.currentTimeMillis();
                                if (isSystemActive && (isReading || isRestroomMode) && (now - lastFrameTime >= FRAME_INTERVAL)) {
                                    lastFrameTime = now;

                                    // Mandamos el frame al hilo de IA para no colgar la recepción de datos de red
                                    aiExecutor.execute(() -> {
                                        if (isReading) {
                                            analyzeOcr(rotatedBitmap);
                                        } else if (isRestroomMode) {
                                            analyzeYolo(rotatedBitmap);
                                        }
                                    });
                                }
                            }
                            jpegBuffer.reset();
                            isCaptureStarted = false;
                        }
                    } else {
                        if (prevByte == 0xFF && currentByte == 0xD8) {
                            isCaptureStarted = true;
                            jpegBuffer.write(prevByte);
                            jpegBuffer.write(currentByte);
                        }
                    }
                    prevByte = currentByte;
                }
            } catch (Exception e) {
                Log.e(TAG, "Error en flujo del Stream ESP32: " + e.getMessage());
            } finally {
                try { if (bis != null) bis.close(); if (connection != null) connection.disconnect(); } catch (IOException ignored) {}
            }
        });
    }

    private Bitmap rotateBitmap(Bitmap source, float angle) {
        if (angle == 0) return source;
        Matrix matrix = new Matrix();
        matrix.postRotate(angle);
        Bitmap rotated = Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
        if (rotated != source) source.recycle();
        return rotated;
    }

    // ════════════════════════════════════════════════════════════════════════
    // Analizador OCR (ML Kit) Adaptado a Bitmap
    // ════════════════════════════════════════════════════════════════════════

    private void analyzeOcr(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int cropWidth = (int) (width * 0.707);
        int cropHeight = (int) (height * 0.707);
        int left = (width - cropWidth) / 2;
        int top = (height - cropHeight) / 2;

        Bitmap cropped = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight);

        textRecognizer.process(InputImage.fromBitmap(cropped, 0))
                .addOnSuccessListener(result -> {
                    String clean = normalize(result.getText());
                    if (!clean.isEmpty()) {
                        runOnUiThread(() -> { if (tvDetected != null) tvDetected.setText(clean); });
                        detectAndHandleOcr(clean);
                    }
                })
                .addOnCompleteListener(t -> cropped.recycle()); // Cuidamos la memoria RAM
    }

    private void detectAndHandleOcr(String text) {
        if (text == null || text.trim().isEmpty()) return;
        long now = System.currentTimeMillis();
        String speech;

        if (text.matches(".*S[A4]L[I1]D[A4].*[E3]M[E3]RG[E3]NC[I1][A4].*")) speech = "Salida de emergencia";
        else if (text.matches(".*[E3]XT[I1]NT[O0]R.*")) speech = "Extintor";
        else if (text.matches(".*[A4]L[A4]RM[A4].*")) speech = "Alarma";
        else if (text.matches(".*R[U0]T[A4].*[E3]V[A4]C[U0][A4]C[I1][O0]N.*")) speech = "Ruta de evacuación";
        else if (text.matches(".*S[A4]L[I1]D[A4].*")) speech = "Salida";
        else if (text.matches(".*[I1]N[O0]RG[A4]N[I1]C[O0].*")) speech = "Inorgánico";
        else if (text.matches(".*[O0]RG[A4]N[I1]C[O0].*")) speech = "Orgánico";
        else if (text.matches(".*PL[A4]ST[I1]C[O0].*")) speech = "Plástico";
        else if (text.matches(".*M[E3]T[A4]L.*")) speech = "Metal";
        else speech = text.toLowerCase(Locale.ROOT);

        if (speech.equals(lastScene)) stableCount++;
        else { stableCount = 1; lastScene = speech; }

        if (stableCount >= STABLE_THRESHOLD && (now - lastSceneTime) > SCENE_COOLDOWN) {
            lastSceneTime = now;
            speak(speech);
            stableCount = 0;
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Analizador YOLO (TFLite) Adaptado a Bitmap
    // ════════════════════════════════════════════════════════════════════════

    private void analyzeYolo(Bitmap bitmap) {
        if (!yoloReady || yoloDetector == null) return;

        try {
            long t0 = System.currentTimeMillis();
            List<DetectionResult> results = yoloDetector.detect(bitmap);
            long elapsed = System.currentTimeMillis() - t0;

            runOnUiThread(() -> {
                if (overlayView != null) overlayView.setResults(results);
                if (tvInference != null) {
                    tvInference.setText(String.format(Locale.US,
                            "Inferencia: %d ms | Objetos: %d", elapsed, results.size()));
                }
            });

            announceYoloDetection(results);
        } catch (Exception e) {
            Log.e(TAG, "Error procesando frame en YOLO", e);
        }
    }

    private void announceYoloDetection(List<DetectionResult> results) {
        if (results.isEmpty()) return;
        long now = System.currentTimeMillis();

        for (DetectionResult det : results) {
            if (det.getConfidencePercent() >= 60f) {
                String label = det.getClassName().toLowerCase(Locale.ROOT);
                if (label.equals(lastYoloScene) && (now - lastYoloSceneTime) < YOLO_COOLDOWN) continue;

                lastYoloScene     = label;
                lastYoloSceneTime = now;
                String frase;

                if (isRestroomMode || label.contains("baño") || label.contains("bano") || label.contains("toilet")) {
                    frase = "Hay un baño de frente";
                } else {
                    frase = "Detectado: " + det.getClassName();
                }
                speak(frase);
                break;
            }
        }
    }

    // ── Motores y Cargas ─────────────────────────────────────────────────────

    private void loadYoloModel() {
        aiExecutor.execute(() -> {
            try {
                yoloDetector = new YoloDetector(this, classNames);
                yoloReady    = true;
                Log.d(TAG, "Modelo YOLO cargado exitosamente.");
            } catch (IOException e) {
                Log.e(TAG, "Error fatal al cargar modelo .tflite", e);
            }
        });
    }

    private String[] loadLabels() {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) labels.add(line.trim());
            }
        } catch (IOException e) {
            labels.add("baño");
        }
        return labels.toArray(new String[0]);
    }

    private void requestPermissionsIfNeeded() {
        // Ya no pedimos permiso de cámara física del cel, solo Micrófono
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 100);
        } else {
            startMjpegStream();
            speechRecognizer.startListening(speechRecognizerIntent);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == 100 && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startMjpegStream();
            speechRecognizer.startListening(speechRecognizerIntent);
        }
    }

    private String normalize(String t) {
        if (t == null) return "";
        return Normalizer.normalize(t, Normalizer.Form.NFD).replaceAll("\\p{M}", "").replace("\n", " ").replaceAll("\\s+", " ").trim().toUpperCase(Locale.ROOT);
    }

    private void restartListening() {
        handler.postDelayed(() -> {
            if (!isListening) {
                try {
                    speechRecognizer.startListening(speechRecognizerIntent);
                } catch (Exception e) {
                    Log.e(TAG, "Fallo al reactivar STT", e);
                }
            }
        }, 700);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        isStreaming = false;
        if (tts              != null) { tts.shutdown();            tts              = null; }
        if (speechRecognizer != null) { speechRecognizer.destroy(); speechRecognizer = null; }
        if (yoloDetector     != null) { yoloDetector.close();      yoloDetector     = null; }
        streamExecutor.shutdown();
        aiExecutor.shutdown();
    }
}