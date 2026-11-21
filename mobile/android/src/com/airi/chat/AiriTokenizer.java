package com.airi.chat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class AiriTokenizer {
    private static final int BYTE_OFFSET = 8;
    private static final int FIRST_MERGE = BYTE_OFFSET + 256;
    static final int EOS = 2;

    private final int[][] merges;
    private final byte[][] vocabulary;

    AiriTokenizer(JSONObject config) throws JSONException {
        String version = config.getString("version");
        JSONArray rows;
        switch (version) {
            case "byte-v1": rows = new JSONArray(); break;
            case "bpe-v1": rows = config.getJSONArray("merges"); break;
            default: throw new JSONException("Versione del tokenizer incompatibile");
        }
        if (rows.length() > 4096 || config.getInt("vocab_size") != FIRST_MERGE + rows.length()) {
            throw new JSONException("Dimensione del vocabolario non valida");
        }
        merges = new int[rows.length()][2];
        vocabulary = new byte[FIRST_MERGE + rows.length()][];
        for (int value = 0; value < 256; value++) vocabulary[BYTE_OFFSET + value] = new byte[]{(byte) value};
        for (int index = 0; index < rows.length(); index++) {
            int token = FIRST_MERGE + index;
            JSONArray pair = rows.getJSONArray(index);
            int left = pair.getInt(0);
            int right = pair.getInt(1);
            if (pair.length() != 2 || left < BYTE_OFFSET || right < BYTE_OFFSET || left >= token || right >= token) {
                throw new JSONException("Regola del tokenizer non valida");
            }
            merges[index][0] = left;
            merges[index][1] = right;
            byte[] prefix = vocabulary[left];
            byte[] suffix = vocabulary[right];
            if (prefix.length + suffix.length > 65536) throw new JSONException("Token troppo lungo");
            vocabulary[token] = Arrays.copyOf(prefix, prefix.length + suffix.length);
            System.arraycopy(suffix, 0, vocabulary[token], prefix.length, suffix.length);
        }
    }

    int size() { return vocabulary.length; }

    List<Integer> prompt(JSONArray messages, int contextLength) throws JSONException {
        ArrayList<Integer> prompt = new ArrayList<>();
        for (int index = messages.length() - 1; index >= 0 && prompt.size() < contextLength; index--) {
            JSONObject message = messages.getJSONObject(index);
            ArrayList<Integer> turn = encode(message.getString("content"));
            String role = message.getString("role");
            turn.add(0, role.equals("user") ? 4 : 5);
            turn.add(7);
            prompt.addAll(0, turn);
        }
        prompt.add(0, 1);
        prompt.add(5);
        return new ArrayList<>(prompt.subList(Math.max(0, prompt.size() - contextLength), prompt.size()));
    }

    private ArrayList<Integer> encode(String text) {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        int[] tokens = new int[bytes.length];
        int size = tokens.length;
        for (int index = 0; index < size; index++) tokens[index] = BYTE_OFFSET + (bytes[index] & 0xff);
        for (int index = 0; index < merges.length; index++) {
            int left = merges[index][0], right = merges[index][1];
            int write = 0;
            for (int read = 0; read < size; read++) {
                int token = tokens[read];
                if (token == left && read + 1 < size && tokens[read + 1] == right) {
                    token = FIRST_MERGE + index;
                    read++;
                }
                tokens[write++] = token;
            }
            size = write;
        }
        ArrayList<Integer> result = new ArrayList<>(size + 2);
        for (int index = 0; index < size; index++) result.add(tokens[index]);
        return result;
    }

    String decode(List<Integer> tokens) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        for (int token : tokens) {
            if (token >= BYTE_OFFSET) {
                byte[] bytes = vocabulary[token];
                output.write(bytes, 0, bytes.length);
            }
        }
        return new String(output.toByteArray(), StandardCharsets.UTF_8);
    }
}
