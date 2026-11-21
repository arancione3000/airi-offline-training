package com.airi.chat;

import android.app.Activity;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
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
    private final ExecutorService updates = Executors.newSingleThreadExecutor();
    private final Handler clock = new Handler(Looper.getMainLooper());
    private final Runnable checkForUpdate = new Runnable() {
        @Override public void run() {
            prepareModel();
            clock.postDelayed(this, 30 * 60 * 1000L);
        }
    };
    private final ArrayList<JSONObject> history = new ArrayList<>();
    private AiriModels models;
    private LinearLayout feed;
    private ScrollView scroll;
    private TextView status;
    private EditText input;
    private Button send;
    private Button clear;
    private Button retry;
    private boolean busy;
    private boolean updating;

    @Override public void onCreate(Bundle saved) {
        super.onCreate(saved);
        getWindow().setStatusBarColor(BACKGROUND);
        getWindow().setNavigationBarColor(BACKGROUND);
        models = AiriModels.from(this);
        deleteSharedPreferences("connection");
        LinearLayout screen = vertical();
        screen.setBackgroundColor(BACKGROUND);
        screen.setPadding(dp(18), dp(14), dp(18), dp(12));

        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout title = vertical();
        TextView brand = text("Airi", 31, LIGHT);
        brand.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        title.addView(brand);
        status = text("Avvio di Airi", 13, MUTED);
        status.setAccessibilityLiveRegion(View.ACCESSIBILITY_LIVE_REGION_POLITE);
        title.addView(status);
        top.addView(title, new LinearLayout.LayoutParams(0, -2, 1));
        clear = button("Nuova", PANEL, LIGHT);
        retry = button("Riprova", PANEL, LIGHT);
        retry.setVisibility(View.GONE);
        top.addView(clear);
        LinearLayout.LayoutParams retryParams = new LinearLayout.LayoutParams(-2, -2);
        retryParams.leftMargin = dp(6);
        top.addView(retry, retryParams);
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

        retry.setOnClickListener(view -> prepareModel());
        clear.setOnClickListener(view -> { history.clear(); welcome(); });
        input.setOnEditorActionListener((view, action, event) -> {
            if (action == EditorInfo.IME_ACTION_SEND) { sendMessage(); return true; }
            return false;
        });
        send.setOnClickListener(view -> sendMessage());
        welcome();
        updateControls();
    }

    private void welcome() {
        feed.removeAllViews();
        TextView line = text("Scrivi per iniziare la conversazione.", 16, MUTED);
        line.setPadding(dp(8), dp(9), dp(8), dp(9));
        feed.addView(line);
    }

    @Override protected void onStart() {
        super.onStart();
        clock.post(checkForUpdate);
    }

    @Override protected void onStop() {
        clock.removeCallbacks(checkForUpdate);
        super.onStop();
    }

    private void prepareModel() {
        if (updating || isDestroyed()) return;
        updating = true;
        retry.setVisibility(View.GONE);
        if (models.revision() < 0) status.setText("Preparazione di Airi · al primo avvio serve Internet");
        updates.submit(() -> {
            models.restore();
            if (models.revision() >= 0) runOnUiThread(this::showModel);
            try {
                models.refresh();
            } catch (Exception | LinkageError error) {
                Log.w("Airi", "Caricamento del modello non riuscito", error);
            }
            runOnUiThread(() -> {
                if (isDestroyed()) return;
                updating = false;
                if (models.revision() >= 0) showModel();
                else {
                    status.setText("Download non riuscito · controlla la connessione");
                    retry.setVisibility(View.VISIBLE);
                }
            });
        });
    }

    private void showModel() {
        if (isDestroyed()) return;
        if (!busy) status.setText("Sul telefono · revisione " + models.revision());
        updateControls();
    }

    private void updateControls() {
        send.setEnabled(!busy && models.revision() >= 0);
        clear.setEnabled(!busy);
    }

    private void sendMessage() {
        String question = input.getText().toString().trim();
        if (busy || question.isEmpty() || models.revision() < 0) return;
        if (history.isEmpty()) feed.removeAllViews();
        input.setText("");
        bubble(question, true);
        busy = true;
        updateControls();
        status.setText("Airi sta rispondendo");
        JSONArray messages = new JSONArray(history);
        worker.submit(() -> {
            try {
                JSONObject user = message("user", question);
                messages.put(user);
                String answer = models.answer(messages);
                JSONObject reply = message("assistant", answer);
                runOnUiThread(() -> {
                    if (isDestroyed()) return;
                    if (answer.isEmpty()) {
                        bubble("Airi ha restituito una risposta vuota.", false);
                    } else {
                        history.add(user);
                        history.add(reply);
                        if (history.size() > 46) {
                            history.remove(0);
                            history.remove(0);
                        }
                        bubble(answer, false);
                    }
                    busy = false;
                    showModel();
                });
            } catch (Exception | LinkageError error) {
                Log.w("Airi", "Not able to generate a reply", error);
                runOnUiThread(() -> {
                    if (isDestroyed()) return;
                    bubble("Airi non è riuscita a rispondere. Riprova.", false);
                    if (input.length() == 0) input.setText(question);
                    busy = false;
                    showModel();
                });
            }
        });
    }

    private static JSONObject message(String role, String value) throws Exception {
        return new JSONObject().put("role", role).put("content", value);
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
        updates.shutdownNow();
        super.onDestroy();
    }
}
