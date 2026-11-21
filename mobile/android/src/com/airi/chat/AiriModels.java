package com.airi.chat;

import android.content.Context;
import android.os.SystemClock;
import android.util.AtomicFile;
import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Locale;

final class AiriModels {
    private static final String PROJECT = "arancione3000/airi-offline-training";
    private static final String POINTER = "https://raw.githubusercontent.com/" + PROJECT
            + "/airi-android-current/models/AIRI_ANDROID_CURRENT.json";
    private static final String RELEASES = "https://github.com/" + PROJECT + "/releases/download/";
    private static final long MAX_MODEL_BYTES = 512L * 1024 * 1024;
    private static final long UPDATE_INTERVAL_MS = 30 * 60 * 1000;
    private static AiriModels instance;

    static synchronized AiriModels from(Context context) {
        if (instance == null) instance = new AiriModels(new File(context.getFilesDir(), "airi-models"));
        return instance;
    }

    private final File directory;
    private final AtomicFile selected;
    private final Object inference = new Object();
    private volatile AiriLocalEngine current;
    private long lastCheck = -UPDATE_INTERVAL_MS;

    private AiriModels(File directory) {
        this.directory = directory;
        selected = new AtomicFile(new File(directory, "current.json"));
    }

    int revision() {
        AiriLocalEngine engine = current;
        return engine == null ? -1 : engine.revision;
    }

    String answer(JSONArray messages) throws Exception {
        synchronized (inference) {
            if (current == null) throw new IOException("Il modello non è ancora disponibile");
            return current.answer(messages);
        }
    }

    synchronized void restore() {
        if (current != null) return;
        try {
            JSONObject record = readRecord(selected.readFully());
            File file = modelFile(record);
            if (file.length() != record.getLong("bytes")) throw new IOException("Modello locale incompleto");
            activate(new AiriLocalEngine(file, record));
        } catch (FileNotFoundException firstLaunch) {
            // The app has not installed a model yet.
        } catch (Exception | LinkageError error) {
            lastCheck = -UPDATE_INTERVAL_MS;
            Log.w("Airi", "Impossibile aprire il modello locale", error);
        }
    }

    synchronized void refresh() throws Exception {
        if (revision() >= 0 && SystemClock.elapsedRealtime() - lastCheck < UPDATE_INTERVAL_MS) return;
        lastCheck = SystemClock.elapsedRealtime();
        JSONObject record = fetchRecord();
        int revision = record.getInt("revision");
        if (revision <= revision()) return;
        if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Spazio locale non disponibile");
        File target = modelFile(record);
        File temporary = File.createTempFile("download-", ".onnx", directory);
        AiriLocalEngine candidate = null;
        try {
            download(record, temporary);
            candidate = new AiriLocalEngine(temporary, record);
            if (!temporary.renameTo(target)) throw failure("Salvataggio del modello non riuscito");
            writeRecord(record);
            activate(candidate);
            candidate = null;
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (!file.equals(target) && file.getName().endsWith(".onnx")) file.delete();
                }
            }
        } finally {
            if (candidate != null) candidate.close();
            temporary.delete();
        }
    }

    private void activate(AiriLocalEngine candidate) {
        synchronized (inference) {
            AiriLocalEngine old = current;
            current = candidate;
            if (old != null) {
                try { old.close(); } catch (Exception error) { Log.w("Airi", "Chiusura del modello precedente", error); }
            }
        }
    }

    private void writeRecord(JSONObject record) throws IOException {
        FileOutputStream out = selected.startWrite();
        try {
            out.write(record.toString().getBytes(StandardCharsets.UTF_8));
            selected.finishWrite(out);
        } catch (IOException error) {
            selected.failWrite(out);
            throw error;
        }
    }

    private File modelFile(JSONObject record) throws JSONException {
        return new File(directory, "airi-" + record.getInt("revision") + "-" + record.getString("sha256") + ".onnx");
    }

    private static JSONObject fetchRecord() throws Exception {
        HttpURLConnection connection = connect(POINTER);
        try (InputStream stream = connection.getInputStream();
             ByteArrayOutputStream bytes = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int count;
            while ((count = stream.read(buffer)) != -1) {
                if (bytes.size() + count > 256 * 1024) throw failure("Indice del modello troppo grande");
                bytes.write(buffer, 0, count);
            }
            return readRecord(bytes.toByteArray());
        } finally {
            connection.disconnect();
        }
    }

    private static JSONObject readRecord(byte[] bytes) throws JSONException, IOException {
        JSONObject record = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
        int context = record.getInt("context_length");
        long size = record.getLong("bytes");
        int revision = record.getInt("revision");
        String sha = record.getString("sha256");
        if (record.getInt("schema") != 1 || revision < 0 || context < 4 || context > 8192
                || !sha.matches("[0-9a-f]{64}") || size <= 0 || size > MAX_MODEL_BYTES) {
            throw failure("Versione del modello incompatibile");
        }
        return record;
    }

    private static void download(JSONObject record, File output) throws Exception {
        String sha = record.getString("sha256");
        long expectedSize = record.getLong("bytes");
        String tag = "airi-android-model-r" + record.getInt("revision") + "-" + sha.substring(0, 12);
        HttpURLConnection connection = connect(RELEASES + tag + "/model.onnx");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream stream = connection.getInputStream(); FileOutputStream file = new FileOutputStream(output)) {
            long reportedSize = connection.getContentLengthLong();
            if (reportedSize > 0 && reportedSize != expectedSize) throw failure("Dimensione del modello errata");
            byte[] buffer = new byte[128 * 1024];
            long size = 0;
            int count;
            while ((count = stream.read(buffer)) != -1) {
                if (Thread.currentThread().isInterrupted()) throw failure("Download interrotto");
                size += count;
                if (size > expectedSize) throw failure("Dimensione del modello errata");
                digest.update(buffer, 0, count);
                file.write(buffer, 0, count);
            }
            StringBuilder actual = new StringBuilder();
            for (byte value : digest.digest()) actual.append(String.format(Locale.ROOT, "%02x", value & 0xff));
            if (size != expectedSize || !sha.contentEquals(actual)) throw failure("Verifica del modello non riuscita");
            file.getFD().sync();
        } finally {
            connection.disconnect();
        }
    }

    private static HttpURLConnection connect(String address) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) URI.create(address).toURL().openConnection();
        connection.setConnectTimeout(12000);
        connection.setReadTimeout(30000);
        connection.setUseCaches(false);
        connection.setRequestProperty("Accept-Encoding", "identity");
        return connection;
    }

    private static IOException failure(String message) { return new IOException(message); }
}
