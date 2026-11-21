package com.airi.chat;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

final class AiriLocalEngine implements AutoCloseable {
    private static final int MAX_CONTEXT_TOKENS = 256;
    private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final AiriTokenizer tokenizer;
    private final int contextLength;
    final int revision;

    AiriLocalEngine(File model, JSONObject manifest) throws JSONException, OrtException {
        revision = manifest.getInt("revision");
        contextLength = Math.min(MAX_CONTEXT_TOKENS, manifest.getInt("context_length"));
        tokenizer = new AiriTokenizer(manifest.getJSONObject("tokenizer"));
        try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
            options.setIntraOpNumThreads(Math.min(4, Runtime.getRuntime().availableProcessors()));
            // Lengths vary per token. Avoid accumulating an arena sized for earlier prompts.
            options.setCPUArenaAllocator(false);
            session = environment.createSession(model.toString(), options);
        }
        try {
            nextToken(Collections.singletonList(1));
        } catch (OrtException | RuntimeException invalid) {
            session.close();
            throw invalid;
        }
    }

    String answer(JSONArray messages) throws JSONException, OrtException {
        List<Integer> prompt = tokenizer.prompt(messages, contextLength);
        List<Integer> generated = new ArrayList<>();
        for (int step = 0; step < 96 && !Thread.currentThread().isInterrupted(); step++) {
            int token = nextToken(prompt.subList(Math.max(0, prompt.size() - contextLength), prompt.size()));
            if (token == AiriTokenizer.EOS) break;
            generated.add(token);
            prompt.add(token);
        }
        return tokenizer.decode(generated).trim();
    }

    private int nextToken(List<Integer> context) throws OrtException {
        long[] input = new long[context.size()];
        for (int index = 0; index < context.size(); index++) input[index] = context.get(index);
        try (OnnxTensor tensor = OnnxTensor.createTensor(environment, LongBuffer.wrap(input), new long[]{1, input.length});
             OrtSession.Result result = session.run(Collections.singletonMap("input_ids", tensor))) {
            float[] logits = ((float[][]) result.get(0).getValue())[0];
            if (logits.length != tokenizer.size()) throw new IllegalArgumentException("Vocabolario incompatibile");
            int best = 0;
            for (int index = 1; index < logits.length; index++) {
                if (logits[index] > logits[best]) best = index;
            }
            return best;
        }
    }

    @Override public void close() throws OrtException { session.close(); }
}
