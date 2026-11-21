package com.airi.chat;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import org.json.JSONArray;
import org.json.JSONObject;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutorService;

public final class MainActivity extends Activity {
    private static final int BACKGROUND = Color.rgb(13, 23, 30);
    private static final int PANEL = Color.rgb(24, 42, 51);
    private static final int LIGHT = Color.rgb(231, 243, 241);
    private static final int MUTED = Color.rgb(165, 188, 190);
    private static final int ACCENT = Color.rgb(128, 228, 202);
    private static final int OWN = Color.rgb(36, 80, 78);

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final ArrayList<JSONObject> history = new ArrayList<>();
    private SharedPreferences preferences;
    private LinearLayout feed;
    private ScrollView scroll;
    private TextView status;
    private EditText input;
    private Button send;
    private Button clear;
    private Button settings;
    private boolean busy;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);
        preferences = getSharedPreferences("connection", MODE_PRIVATE);
        LinearLayout screen = vertical();
        screen.setBackgroundColor(BACKGROUND);
        screen.setPadding(dp(18), dp(14), dp(18), dp(12));

        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout title = vertical();
        TextView brand = text("Airi", 31, LIGHT);
        brand.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.addView(brand);
        status = text("Connessione", 13, MUTED);
        status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        title.addView(status);
        top.addView(title, new LinearLayout.LayoutParams(0, -2, 1));
        clear = button("Nuova", PANEL, LIGHT);
        settings = button("Server", PANEL, LIGHT);
        top.addView(clear);
        LinearLayout.LayoutParams settingsParams = new LinearLayout.LayoutParams(-2, -2);
        settingsParams.leftMargin = dp(6);
        top.addView(settings, settingsParams);
        screen.addView(top);

        scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        feed = vertical();
        feed.setPadding(0, dp(20), 0, dp(10));
        scroll.addView(feed);
        screen.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));

        LinearLayout composer = new LinearLayout(this);
        composer.setGravity(Gravity.CENTER_VERTICAL);
        input = new EditText(this);
        input.setTextColor(LIGHT);
        input.setHintTextColor(MUTED);
        input.setHint("Messaggio per Airi");
        input.setTextSize(16);
        input.setPadding(dp(14), dp(10), dp(14), dp(10));
        input.setBackground(round(PANEL, 16));
        input.setMaxLines(4);
        input.setFilters(new android.text.InputFilter[]{new android.text.InputFilter.LengthFilter(4096)});
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        composer.addView(input, new LinearLayout.LayoutParams(0, -2, 1));
        send = button("Invia", ACCENT, BACKGROUND);
        LinearLayout.LayoutParams sendParams = new LinearLayout.LayoutParams(-2, -2);
        sendParams.leftMargin = dp(8);
        composer.addView(send, sendParams);
        screen.addView(composer);
        setContentView(screen);

        settings.setOnClickListener(view -> showSettings());
        clear.setOnClickListener(view -> { history.clear(); welcome(); });
        input.setOnEditorActionListener((view, action, event) -> {
            if (action == EditorInfo.IME_ACTION_SEND) { sendMessage(); return true; }
            return false;
        });
        send.setOnClickListener(view -> sendMessage());
        welcome();
        if (address().isEmpty()) showSettings();
        else checkConnection();
    }

    private void welcome() {
        feed.removeAllViews();
        TextView line = text(address().isEmpty() ? "Collega il server di Airi per iniziare." : "Scrivi per iniziare la conversazione.", 16, MUTED);
        line.setPadding(dp(8), dp(9), dp(8), dp(9));
        feed.addView(line);
    }

    private String address() { return preferences.getString("address", ""); }
    private String token() { return preferences.getString("token", ""); }

    private void showSettings() {
        LinearLayout layout = vertical();
        layout.setPadding(dp(22), dp(12), dp(22), 0);
        EditText addressInput = new EditText(this);
        addressInput.setSingleLine();
        addressInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        addressInput.setHint("https:// oppure IP locale");
        addressInput.setText(address());
        layout.addView(text("Indirizzo del server", 14, MUTED));
        layout.addView(addressInput);
        EditText tokenInput = new EditText(this);
        tokenInput.setSingleLine();
        tokenInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        tokenInput.setText(token());
        layout.addView(text("Chiave di accesso", 14, MUTED));
        layout.addView(tokenInput);
        AlertDialog dialog = new AlertDialog.Builder(this).setTitle("Server Airi").setView(layout)
                .setNegativeButton("Annulla", null).setPositiveButton("Collega", null).create();
        dialog.setOnShowListener(unused -> dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(view -> {
            String next = addressInput.getText().toString().trim();
            String key = tokenInput.getText().toString().trim();
            try {
                new AiriConnection(next, key);
            } catch (IllegalArgumentException invalid) {
                addressInput.setError(invalid.getMessage());
                return;
            }
            if (!next.equals(address()) || !key.equals(token())) {
                history.clear();
            }
            preferences.edit().putString("address", next).putString("token", key).apply();
            if (history.isEmpty()) welcome();
            dialog.dismiss();
            checkConnection();
        }));
        dialog.show();
        if (address().isEmpty()) status.setText("Server da configurare");
    }

    private void checkConnection() {
        String host = address(), key = token();
        status.setText("Connessione in corso");
        worker.submit(() -> {
            try {
                JSONObject result = new AiriConnection(host, key).request("/v1/status", null);
                if (result.optInt("api_version") != 1) throw new IllegalArgumentException();
                runOnUiThread(() -> {
                    if (host.equals(address()) && key.equals(token())) {
                        status.setText(result.isNull("model_revision") ? "Server raggiunto · modello in caricamento"
                                : "Connessa · revisione " + result.optInt("model_revision"));
                    }
                });
            } catch (Exception failed) {
                runOnUiThread(() -> { if (host.equals(address()) && key.equals(token())) status.setText("Server non raggiungibile o chiave errata"); });
            }
        });
    }

    private void sendMessage() {
        String question = input.getText().toString().trim();
        if (busy || question.isEmpty()) return;
        if (address().isEmpty()) { showSettings(); return; }
        if (history.isEmpty()) feed.removeAllViews();
        String host = address(), key = token();
        input.setText("");
        bubble(question, true);
        busy = true;
        send.setEnabled(false); clear.setEnabled(false); settings.setEnabled(false);
        status.setText("Airi sta rispondendo");
        worker.submit(() -> {
            try {
                JSONObject request = new JSONObject();
                JSONArray messages = new JSONArray();
                int start = Math.max(0, history.size() - 46);
                for (int i = start; i < history.size(); i++) messages.put(history.get(i));
                JSONObject user = message("user", question);
                messages.put(user);
                request.put("messages", messages);
                while (request.toString().getBytes(StandardCharsets.UTF_8).length > 64000 && messages.length() > 1) {
                    messages.remove(0); messages.remove(0);
                }
                JSONObject response = new AiriConnection(host, key).request("/v1/chat", request);
                String answer = response.getString("answer");
                JSONObject reply = message("assistant", answer);
                runOnUiThread(() -> {
                    history.add(user); history.add(reply);
                    if (history.size() > 46) { history.remove(0); history.remove(0); }
                    if (answer.trim().isEmpty()) {
                        history.remove(history.size() - 1); history.remove(history.size() - 1);
                        bubble("Airi ha restituito una risposta vuota.", false);
                    } else bubble(answer, false);
                    status.setText(response.isNull("model_revision") ? "Connessa"
                            : "Connessa · revisione " + response.optInt("model_revision"));
                    ready();
                });
            } catch (Exception error) {
                runOnUiThread(() -> {
                    String message = error instanceof java.io.IOException && error.getMessage() != null
                            && (error.getMessage().startsWith("Airi") || error.getMessage().startsWith("Chiave")
                                || error.getMessage().startsWith("Il") || error.getMessage().startsWith("L’indirizzo"))
                            ? error.getMessage() : "Connessione interrotta. Controlla il server e riprova.";
                    bubble(message, false);
                    if (input.length() == 0) input.setText(question);
                    status.setText("Invio non riuscito");
                    ready();
                });
            }
        });
    }

    private static JSONObject message(String role, String value) throws Exception {
        return new JSONObject().put("role", role).put("content", value);
    }

    private void ready() {
        busy = false; send.setEnabled(true); clear.setEnabled(true); settings.setEnabled(true);
    }

    private void bubble(String value, boolean own) {
        FrameLayout row = new FrameLayout(this);
        TextView label = text(value, 16, LIGHT);
        label.setTextIsSelectable(true);
        label.setPadding(dp(15), dp(11), dp(15), dp(11));
        label.setMaxWidth(Math.min(dp(460), getResources().getDisplayMetrics().widthPixels - dp(80)));
        label.setBackground(round(own ? OWN : PANEL, 17));
        FrameLayout.LayoutParams inside = new FrameLayout.LayoutParams(-2, -2, own ? Gravity.END : Gravity.START);
        row.addView(label, inside);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.bottomMargin = dp(11);
        feed.addView(row, params);
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private LinearLayout vertical() {
        LinearLayout view = new LinearLayout(this);
        view.setOrientation(LinearLayout.VERTICAL);
        return view;
    }

    private TextView text(String label, int size, int color) {
        TextView view = new TextView(this);
        view.setText(label); view.setTextSize(size); view.setTextColor(color);
        return view;
    }

    private Button button(String label, int background, int foreground) {
        Button view = new Button(this);
        view.setText(label);
        view.setAllCaps(false);
        view.setTextColor(foreground);
        view.setMinimumHeight(dp(48));
        view.setMinWidth(dp(56));
        view.setPadding(dp(13), dp(7), dp(13), dp(7));
        view.setBackgroundTintList(ColorStateList.valueOf(background));
        return view;
    }

    private GradientDrawable round(int color, int radius) {
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(color); shape.setCornerRadius(dp(radius));
        return shape;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override protected void onDestroy() {
        worker.shutdownNow();
        super.onDestroy();
    }
}
