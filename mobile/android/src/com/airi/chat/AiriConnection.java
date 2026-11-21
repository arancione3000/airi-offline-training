package com.airi.chat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import org.json.JSONObject;

final class AiriConnection {
    private final String base;
    private final String token;

    AiriConnection(String address, String token) {
        URI uri = URI.create(address.trim());
        String host = uri.getHost();
        String scheme = uri.getScheme();
        if (host == null || uri.getUserInfo() != null || uri.getFragment() != null || uri.getQuery() != null
                || (!"https".equals(scheme) && !("http".equals(scheme) && local(host)))) {
            throw new IllegalArgumentException("Usa HTTPS oppure un indirizzo IP della rete locale con HTTP.");
        }
        if (!token.matches("[!-~]{24,256}")) {
            throw new IllegalArgumentException("Inserisci la chiave fornita dal server.");
        }
        this.base = address.trim().replaceAll("/+$", "");
        this.token = token;
    }

    private static boolean local(String host) {
        if ("localhost".equals(host) || "[::1]".equals(host)) return true;
        if (!host.matches("[0-9]{1,3}(\\.[0-9]{1,3}){3}")) return false;
        String[] parts = host.split("\\.");
        for (String part : parts) if (Integer.parseInt(part) > 255) return false;
        int a = Integer.parseInt(parts[0]);
        int b = Integer.parseInt(parts[1]);
        return a == 10 || a == 127 || (a == 172 && b >= 16 && b <= 31) || (a == 192 && b == 168);
    }

    JSONObject request(String path, JSONObject payload) throws Exception {
        HttpURLConnection connection = (HttpURLConnection) URI.create(base + path).toURL().openConnection();
        connection.setInstanceFollowRedirects(false);
        connection.setConnectTimeout(12000);
        connection.setReadTimeout(75000);
        connection.setRequestProperty("Authorization", "Bearer " + token);
        connection.setRequestProperty("Accept", "application/json");
        try {
            if (payload != null) {
                byte[] body = payload.toString().getBytes(StandardCharsets.UTF_8);
                connection.setRequestMethod("POST");
                connection.setDoOutput(true);
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");
                connection.setFixedLengthStreamingMode(body.length);
                try (java.io.OutputStream output = connection.getOutputStream()) {
                    output.write(body);
                }
            }
            int status = connection.getResponseCode();
            if (status >= 300 && status < 400) throw new IOException("Il server ha restituito un reindirizzamento.");
            boolean success = status >= 200 && status < 300;
            InputStream stream = success ? connection.getInputStream() : connection.getErrorStream();
            if (stream == null || connection.getContentType() == null
                    || !connection.getContentType().startsWith("application/json")) {
                throw new IOException("L’indirizzo non risponde come server Airi.");
            }
            ByteArrayOutputStream data = new ByteArrayOutputStream();
            try (InputStream input = stream) {
                byte[] buffer = new byte[4096];
                int size;
                while ((size = input.read(buffer)) != -1) {
                    if (data.size() + size > 65536) throw new IOException("Risposta troppo grande.");
                    data.write(buffer, 0, size);
                }
            }
            JSONObject result = new JSONObject(new String(data.toByteArray(), StandardCharsets.UTF_8));
            if (!success) throw new IOException(result.optString("error", "Errore del server Airi."));
            return result;
        } finally {
            connection.disconnect();
        }
    }
}
