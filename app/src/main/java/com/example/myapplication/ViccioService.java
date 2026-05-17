package com.example.myapplication;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Binder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.util.Log;

import androidx.core.app.NotificationCompat;

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

public class ViccioService extends Service {
    private static final String TAG = "ViccioService";
    private static final String CHANNEL_ID = "ViccioServiceChannel";
    private static final String STREAM_URL = "http://10.50.172.39:81/stream";

    private final IBinder binder = new LocalBinder();
    private ServiceCallback callback;

    // Motores e hilos de procesamiento
    private ExecutorService analysisExecutor;
    private TextRecognizer textRecognizer;
    private YoloDetector yoloDetector;
    private TextToSpeech tts;
    private SpeechRecognizer speechRecognizer;
    private Intent speechRecognizerIntent;

    // Estados de la máquina de control por voz
    private boolean isStreaming = false;
    private boolean yoloReady = false;
    private boolean isTtsReady = false;
    private boolean isSpeaking = false;
    private boolean isListening = false;
    private boolean isSystemActive = false;
    private boolean isReading = false;
    private boolean isRestroomMode = false;

    // Parámetros de configuración y estabilidad
    private float cameraRotationAngle = 0.0f; // Modifica aquí si requieres rotación (ej: 90f)
    private int stableCount = 0;
    private static final int STABLE_THRESHOLD = 2;

    private String lastScene = "";
    private long lastSceneTime = 0;
    private static final long SCENE_COOLDOWN = 1500;

    private String lastYoloScene = "";
    private long lastYoloSceneTime = 0;
    private static final long YOLO_COOLDOWN = 3000;

    private String lastWake = "";
    private long lastWakeTime = 0;
    private static final long WAKE_COOLDOWN = 1500;

    private long lastFrameTime = 0;
    private static final long FRAME_INTERVAL = 150;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private String[] classNames;

    public interface ServiceCallback {
        void onFrameProcessed(Bitmap bitmap, List<DetectionResult> results, long inferenceTime, int objectsCount);
        void onOcrTextDetected(String text);
    }

    public class LocalBinder extends Binder {
        ViccioService getService() { return ViccioService.this; }
    }

    @Override
    public IBinder onBind(Intent intent) { return binder; }

    public void setCallback(ServiceCallback callback) { this.callback = callback; }

    @Override
    public void onCreate() {
        super.onCreate();
        analysisExecutor = Executors.newSingleThreadExecutor();
        textRecognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS);

        classNames = loadLabels();
        loadYoloModel();
        setupTTS();
        setupSpeechRecognizer();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        createNotificationChannel();
        Notification notification = new NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Bichio IA Activo")
                .setContentText("Procesando stream de la ESP32 en segundo plano...")
                .setSmallIcon(android.R.drawable.ic_btn_speak_now)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();

        startForeground(1, notification);

        if (!isStreaming) {
            startMjpegStream();
        }
        return START_STICKY;
    }

    // ── CONFIGURACIÓN DEL FLUJO MJPEG HTTP ───────────────────────────────────
    private void startMjpegStream() {
        isStreaming = true;
        analysisExecutor.execute(() -> {
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

                Log.d(TAG, "Conectado al stream HTTP.");

                while (isStreaming && (currentByte = bis.read()) != -1) {
                    if (isCaptureStarted) {
                        jpegBuffer.write(currentByte);
                        if (prevByte == 0xFF && currentByte == 0xD9) {
                            byte[] rawImage = jpegBuffer.toByteArray();
                            Bitmap rawBitmap = BitmapFactory.decodeByteArray(rawImage, 0, rawImage.length);
                            if (rawBitmap != null) {
                                Bitmap rotated = rotateBitmap(rawBitmap, cameraRotationAngle);
                                analyzeFrame(rotated);
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
                Log.e(TAG, "Error en flujo de red ESP32: " + e.getMessage());
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

    private void analyzeFrame(Bitmap bitmap) {
        long now = System.currentTimeMillis();
        List<DetectionResult> currentResults = new ArrayList<>();
        long elapsed = 0;

        // Validar si el procesamiento por IA está encendido y cumple el intervalo de FPS
        if (isSystemActive && (now - lastFrameTime >= FRAME_INTERVAL)) {
            lastFrameTime = now;
            if (isReading) {
                analyzeOcr(bitmap);
            } else if (isRestroomMode) {
                if (yoloReady && yoloDetector != null) {
                    long t0 = System.currentTimeMillis();
                    currentResults = yoloDetector.detect(bitmap);
                    elapsed = System.currentTimeMillis() - t0;
                    announceYoloDetection(currentResults);
                }
            }
        }

        // Si la pantalla de la app está abierta, enviamos la telemetría y el Bitmap
        if (callback != null) {
            final List<DetectionResult> finalResults = currentResults;
            final long finalElapsed = elapsed;
            handler.post(() -> callback.onFrameProcessed(bitmap, finalResults, finalElapsed, finalResults.size()));
        } else {
            // Si la pantalla está oculta, reciclamos el bitmap de inmediato para cuidar la RAM
            bitmap.recycle();
        }
    }

    // ── MÓDULO DE PROCESAMIENTO OCR (ML KIT) ─────────────────────────────────
    private void analyzeOcr(Bitmap bitmap) {
        int width = bitmap.getWidth(), height = bitmap.getHeight();
        int cropWidth = (int) (width * 0.707), cropHeight = (int) (height * 0.707);
        int left = (width - cropWidth) / 2, top = (height - cropHeight) / 2;
        Bitmap cropped = Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight);

        textRecognizer.process(InputImage.fromBitmap(cropped, 0))
                .addOnSuccessListener(result -> {
                    String clean = normalize(result.getText());
                    if (!clean.isEmpty()) {
                        if (callback != null) callback.onOcrTextDetected(clean);
                        detectAndHandleOcr(clean);
                    }
                })
                .addOnCompleteListener(t -> cropped.recycle());
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

    // ── MÓDULO DE PROCESAMIENTO YOLO (TFLITE) ────────────────────────────────
    private void announceYoloDetection(List<DetectionResult> results) {
        if (results.isEmpty()) return;
        long now = System.currentTimeMillis();
        for (DetectionResult det : results) {
            if (det.getConfidencePercent() >= 60f) {
                String label = det.getClassName().toLowerCase(Locale.ROOT);
                if (label.equals(lastYoloScene) && (now - lastYoloSceneTime) < YOLO_COOLDOWN) continue;

                lastYoloScene = label;
                lastYoloSceneTime = now;
                String frase = (isRestroomMode || label.contains("baño") || label.contains("bano") || label.contains("toilet")) ? "Hay un baño de frente" : "Detectado: " + det.getClassName();
                speak(frase);
                break;
            }
        }
    }

    // ── SISTEMA DE VOZ (STT OFFLINE Y TTS) ───────────────────────────────────
    private void setupTTS() {
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(new Locale("es", "MX"));
                isTtsReady = true;
                tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                    @Override public void onStart(String utteranceId) { isSpeaking = true; }
                    @Override public void onDone(String utteranceId) { isSpeaking = false; }
                    @Override public void onError(String utteranceId) { isSpeaking = false; }
                });
                handler.postDelayed(() -> speak("Listo"), 800);
            }
        });
    }

    private void speak(String text) {
        if (tts != null && isTtsReady) tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "BichioId");
    }

    private void setupSpeechRecognizer() {
        handler.post(() -> {
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(getApplicationContext());
            speechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "es-MX");

            // OBLIGATORIO: Forzar reconocimiento local sin requerir datos móviles/internet
            speechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true);

            speechRecognizer.setRecognitionListener(new RecognitionListener() {
                @Override public void onReadyForSpeech(Bundle p) { isListening = true; }
                @Override public void onBeginningOfSpeech() {}
                @Override public void onRmsChanged(float r) {}
                @Override public void onBufferReceived(byte[] b) {}
                @Override public void onEvent(int t, Bundle p) {}
                @Override public void onEndOfSpeech() { isListening = false; restartListening(); }
                @Override public void onError(int error) { isListening = false; restartListening(); }
                @Override public void onResults(Bundle r) { handleVoice(r); restartListening(); }
                @Override public void onPartialResults(Bundle r) { handleVoice(r); }
            });
            speechRecognizer.startListening(speechRecognizerIntent);
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
                isSystemActive = true; lastWake = "INICIAR"; lastWakeTime = now;
                speak("Sistema encendido"); return;
            }
            if (text.contains("LEER") || text.contains("TEXTO") || text.contains("SIGN")) {
                if (!isSystemActive || isReading) return;
                isReading = true; isRestroomMode = false; lastScene = ""; stableCount = 0;
                speak("Lectura de señales activada"); return;
            }
            if (text.contains("BANO") || text.contains("RESTROOM")) {
                if (!isSystemActive || isRestroomMode) return;
                if (!yoloReady) { speak("Modelo no listo"); return; }
                isRestroomMode = true; isReading = false; lastScene = ""; stableCount = 0;
                speak("Búsqueda de baños activada"); return;
            }
            if (text.contains("APAGAR") || text.contains("DETENER")) {
                if (!isSystemActive) return;
                isSystemActive = false; isReading = false; isRestroomMode = false;
                if (tts.isSpeaking()) tts.stop();
                speak("Sistema apagado"); return;
            }
        }
    }

    private void restartListening() {
        handler.postDelayed(() -> {
            if (!isListening && speechRecognizer != null) {
                try { speechRecognizer.startListening(speechRecognizerIntent); } catch (Exception ignored) {}
            }
        }, 700);
    }

    // ── CARGA DE MODELOS Y RECURSOS ASSETS ───────────────────────────────────
    private void loadYoloModel() {
        analysisExecutor.execute(() -> {
            try {
                yoloDetector = new YoloDetector(this, classNames);
                yoloReady = true;
                Log.d(TAG, "YOLO cargado exitosamente en el servicio.");
            } catch (IOException e) { Log.e(TAG, "Error YOLO .tflite", e); }
        });
    }

    private String[] loadLabels() {
        List<String> labels = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(getAssets().open("labels.txt")))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (!line.trim().isEmpty()) labels.add(line.trim());
            }
        } catch (IOException e) { labels.add("baño"); }
        return labels.toArray(new String[0]);
    }

    private String normalize(String t) {
        if (t == null) return "";
        return Normalizer.normalize(t, Normalizer.Form.NFD).replaceAll("\\p{M}", "").replace("\n", " ").replaceAll("\\s+", " ").trim().toUpperCase(Locale.ROOT);
    }

    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(CHANNEL_ID, "Bichio Background Channel", NotificationManager.IMPORTANCE_LOW);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) manager.createNotificationChannel(serviceChannel);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        isStreaming = false;
        if (tts != null) tts.shutdown();
        if (speechRecognizer != null) speechRecognizer.destroy();
        if (yoloDetector != null) yoloDetector.close();
        analysisExecutor.shutdown();
        Log.d(TAG, "Servicio destruido de forma limpia.");
    }

    @Override
    public void onTaskRemoved(Intent rootIntent) {
        super.onTaskRemoved(rootIntent);
        Log.d(TAG, "Usuario eliminó la app de recientes. Destruyendo servicio...");

        // Detiene el stream, apaga los motores y quita la notificación persistente
        isStreaming = false;
        stopSelf();
    }
}