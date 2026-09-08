const $ = (id) => document.getElementById(id);

const ui = {
  backend: $("backend"),
  room: $("room"),
  speaker: $("speaker"),
  language: $("language"),
  microphone: $("microphone"),
  refreshMicsBtn: $("refreshMicsBtn"),
  connectBtn: $("connectBtn"),
  micBtn: $("micBtn"),
  clearBtn: $("clearBtn"),
  copyBtn: $("copyBtn"),
  exportBtn: $("exportBtn"),
  showTranslation: $("showTranslation"),
  connectionBadge: $("connectionBadge"),
  captureBadge: $("captureBadge"),
  versionBadge: $("versionBadge"),
  longTranscript: $("longTranscript"),
  liveTailWrap: $("liveTailWrap"),
  liveTail: $("liveTail"),
  translationWrap: $("translationWrap"),
  translationLabel: $("translationLabel"),
  translationStatus: $("translationStatus"),
  translationText: $("translationText"),
  turnList: $("turnList"),
  segmentTable: $("segmentTable"),
  eventLog: $("eventLog"),
  audioInfo: $("audioInfo"),
  mQueue: $("mQueue"),
  mDropped: $("mDropped"),
  mPreviewErrors: $("mPreviewErrors"),
  mFinalErrors: $("mFinalErrors"),
  mSpeakerResults: $("mSpeakerResults"),
  mSpeakerErrors: $("mSpeakerErrors"),
  mTranslationQueue: $("mTranslationQueue"),
  mTranslationDone: $("mTranslationDone"),
  mTranslationErrors: $("mTranslationErrors"),
  mFinalDecoder: $("mFinalDecoder"),
};

// HLMEET_WEB_V20_FINAL_ONLY_SPEAKER_UI
const state = {
  ws: null,
  connected: false,
  joined: false,
  capturing: false,
  captureReady: false,
  pendingPcmFrames: [],
  pendingPcmBytes: 0,
  audioTransportFault: false,

  // PHASE6G_VI_AUDIO_RELEASE_GUARDS
  captureLanguage: "",
  pcmFlushWaiter: null,

  clientId: "",
  audioContext: null,
  mediaStream: null,
  source: null,
  worklet: null,
  silentGain: null,
  healthTimer: null,

  // segment id -> latest authoritative row
  segments: new Map(),
  // turn id -> Set(segment id)
  turns: new Map(),
  // id -> first-seen order
  order: new Map(),
  nextOrder: 1,

  currentPartial: null,
  // HLMEET_WEB_CONTINUOUS_TRANSLATION_V19_7_1
  // HLMEET_WEB_TRANSLATION_PERSISTENT_DRAFTS_V19_7_1
  // HLMEET_WEB_TRANSLATION_STT_LIKE_V19_7_1_V4
  // HLMEET_WEB_CUMULATIVE_TRANSLATION_V4_2
  //
  // LIVE is only a fast bootstrap preview. Once backend progressive
  // translation exists for a logical turn, that cumulative snapshot owns the
  // display until a newer progressive/FINAL transcript_replace replaces it.
  //
  // Never stitch NMT fragments: target-language word order is not prefix-stable.
  translationPreviews: new Map(),
  health: null,
  eventLines: [],
};

function nowClock() {
  return new Date().toLocaleTimeString("vi-VN", { hour12: false });
}

function logEvent(label, data = null) {
  const suffix = data ? " " + JSON.stringify(data) : "";
  state.eventLines.push(`[${nowClock()}] ${label}${suffix}`);
  if (state.eventLines.length > 300) state.eventLines.splice(0, state.eventLines.length - 300);
  ui.eventLog.textContent = state.eventLines.join("\n");
  ui.eventLog.scrollTop = ui.eventLog.scrollHeight;
}

// 16 kHz * 16-bit mono = 32,000 bytes/s. Keep only a bounded two-second
// transport cushion. This is not browser VAD and never silently discards frames:
// capture is stopped explicitly if the LAN cannot keep up.
const PCM_BYTES_PER_SECOND = 16000 * 2;
const AUDIO_TRANSPORT_BUFFER_MAX_BYTES = PCM_BYTES_PER_SECOND * 2;

function clearPendingPcm() {
  state.pendingPcmFrames = [];
  state.pendingPcmBytes = 0;
}

function stopForAudioTransportFault(reason, detail = null) {
  if (state.audioTransportFault) return;
  state.audioTransportFault = true;
  state.captureReady = false;
  clearPendingPcm();
  logEvent("AUDIO TRANSPORT STOP", { reason, ...(detail || {}) });
  setBadge(ui.captureBadge, "Audio transport stalled", "bad");
  Promise.resolve(stopCapture())
    .catch((err) => logEvent("MIC STOP ERROR", { message: String(err?.message || err) }))
    .finally(() => {
      state.audioTransportFault = false;
      setBadge(ui.captureBadge, "Mic stopped · transport stalled", "bad");
      ui.audioInfo.textContent = "Audio stopped because WebSocket transport could not keep realtime.";
    });
}

function sendPcmFrame(data) {
  if (!state.capturing || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  if (!data || !data.byteLength) return;

  if (!state.captureReady) {
    if (state.pendingPcmBytes + data.byteLength > AUDIO_TRANSPORT_BUFFER_MAX_BYTES) {
      stopForAudioTransportFault("capture-ready-timeout-buffer");
      return;
    }
    state.pendingPcmFrames.push(data);
    state.pendingPcmBytes += data.byteLength;
    return;
  }

  if (state.ws.bufferedAmount > AUDIO_TRANSPORT_BUFFER_MAX_BYTES) {
    stopForAudioTransportFault("websocket-backpressure", {
      bufferedBytes: state.ws.bufferedAmount,
    });
    return;
  }
  state.ws.send(data);
}

function flushPendingPcm() {
  if (!state.captureReady || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const pending = state.pendingPcmFrames;
  clearPendingPcm();
  for (const frame of pending) {
    if (!state.capturing || !state.captureReady) return;
    if (state.ws.bufferedAmount > AUDIO_TRANSPORT_BUFFER_MAX_BYTES) {
      stopForAudioTransportFault("websocket-backpressure", {
        bufferedBytes: state.ws.bufferedAmount,
      });
      return;
    }
    state.ws.send(frame);
  }
}

function backendBase() {
  return ui.backend.value.trim().replace(/\/+$/, "");
}

function wsUrl() {
  const base = backendBase();
  if (!base) throw new Error("Backend URL is empty");
  const u = new URL(base);
  u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
  u.pathname = "/ws";
  u.search = "";
  u.hash = "";
  return u.toString();
}

function setBadge(el, text, kind) {
  el.textContent = text;
  el.className = `badge ${kind}`;
}

function segmentState(row) {
  const status = String(row.verification_status || "");
  if (status.includes("corrected-right-context")) return "corrected";
  if (row.stability === "verified" || row.is_final === true) {
    if (status.includes("unresolved") || status.includes("uncertain")) return "uncertain";
    return "verified";
  }
  if (row.stability === "uncertain") return "uncertain";
  return "provisional";
}

function cleanText(text) {
  return String(text || "").trim().replace(/\s+/g, " ");
}

function speakerLabel(row) {
  return cleanText(row?.speaker_id);
}

function speakerSegments(row) {
  return Array.isArray(row?.speaker_segments) ? row.speaker_segments : [];
}

function punctuatedJoin(parts) {
  // Preserve model punctuation. Add one space between technical segments.
  return parts.map(cleanText).filter(Boolean).join(" ").replace(/\s+([,.;:!?])/g, "$1");
}

function storeTranslationPreview(msg, currentRevision = 0) {
  const turnId = String(msg.turn_id || msg.utterance_id || "");
  const translation = cleanText(msg.translation);
  if (!turnId || !translation) return false;

  const revision = Number(msg.source_revision || 0);
  const previous = state.translationPreviews.get(turnId);
  const previousRevision = Number(previous?.source_revision || 0);

  if (
    revision > 0 &&
    previousRevision > 0 &&
    revision <= previousRevision
  ) {
    return false;
  }

  state.translationPreviews.set(turnId, {
    turn_id: turnId,
    utterance_id: msg.utterance_id || "",
    translation,
    translation_language: msg.translation_language || "",
    source_revision: revision,
    status:
      currentRevision > revision && revision > 0
        ? "translated-live-preview-lagging"
        : (msg.translation_status || "translated-live-preview"),
  });
  return true;
}

function sortedSegments() {
  return [...state.segments.values()].sort((a, b) => {
    const ta = Number(a.started_ms || 0);
    const tb = Number(b.started_ms || 0);
    if (ta !== tb) return ta - tb;
    const oa = state.order.get(a.id) || 0;
    const ob = state.order.get(b.id) || 0;
    return oa - ob;
  });
}

function sortedTurnRows(turnId) {
  const ids = state.turns.get(turnId) || new Set();
  return [...ids]
    .map((id) => state.segments.get(id))
    .filter(Boolean)
    .sort((a, b) => {
      const sa = Number(a.segment_index ?? 0);
      const sb = Number(b.segment_index ?? 0);
      if (sa !== sb) return sa - sb;
      return (state.order.get(a.id) || 0) - (state.order.get(b.id) || 0);
    });
}

function ensureTurn(row) {
  const turnId = row.turn_id || row.id || "unknown";
  if (!state.turns.has(turnId)) state.turns.set(turnId, new Set());
  state.turns.get(turnId).add(row.id);
}

function upsertTranscript(row, isReplace = false) {
  if (!row || !row.id) return;
  const previous = state.segments.get(row.id);
  if (!state.order.has(row.id)) state.order.set(row.id, state.nextOrder++);

  const merged = { ...(previous || {}), ...row };
  state.segments.set(row.id, merged);
  ensureTurn(merged);

  const textChanged = Boolean(
    isReplace &&
    previous &&
    cleanText(previous.text) !== cleanText(merged.text)
  );
  const translationChanged = Boolean(
    isReplace &&
    cleanText(merged.translation) &&
    cleanText(previous?.translation) !== cleanText(merged.translation)
  );

  if (textChanged) {
    merged.__justCorrected = true;
    logEvent("TRANSCRIPT_REPLACE", {
      id: merged.id,
      turn: merged.turn_id,
      seg: merged.segment_index,
      status: merged.verification_status,
      from: previous.text,
      to: merged.text,
    });
    setTimeout(() => {
      const current = state.segments.get(merged.id);
      if (current) {
        delete current.__justCorrected;
        renderAll();
      }
    }, 2600);
  } else {
    logEvent(isReplace ? "TRANSCRIPT_UPDATE" : "TRANSCRIPT", {
      id: merged.id,
      turn: merged.turn_id,
      seg: merged.segment_index,
      state: merged.stability,
      status: merged.verification_status,
      source: merged.asr_source,
      rev: merged.revision,
    });
  }

  if (translationChanged) {
    logEvent("TRANSLATION_UPDATE", {
      id: merged.id,
      turn: merged.turn_id,
      source_language: merged.source_language,
      translation_language: merged.translation_language,
      translation_status: merged.translation_status,
      translation: merged.translation,
    });
  }

  // A rolling/FINAL STT row can replace the browser partial object without
  // clearing translation. The last-good preview or cumulative backend result
  // remains visible until its replacement arrives.
  if (
    state.currentPartial &&
    (state.currentPartial.id === row.id ||
      state.currentPartial.utterance_id === row.id)
  ) {
    state.currentPartial = null;
  }

  // Any backend row translation is cumulative for this logical turn
  // (progressive or FINAL) and therefore outranks disposable LIVE preview.
  if (cleanText(merged.translation)) {
    const turnKey = String(merged.turn_id || merged.id || "");
    if (turnKey) state.translationPreviews.delete(turnKey);
  }

  if (translationChanged) {
    merged.__translationJustCorrected = true;
    setTimeout(() => {
      const current = state.segments.get(merged.id);
      if (current) {
        delete current.__translationJustCorrected;
        renderTranslation();
      }
    }, 2200);
  }

  renderAll();
}
function renderLongTranscript() {
  const rows = sortedSegments();
  if (!rows.length) {
    ui.longTranscript.className = "long-transcript empty";
    ui.longTranscript.innerHTML =
      '<span class="placeholder">Nhấn “Start microphone” và bắt đầu đọc. VIT LIVE hiện realtime; logical FINAL sẽ tự sửa text, sau đó speaker/translation cập nhật metadata.</span>';
    return;
  }

  ui.longTranscript.className = "long-transcript";
  const frag = document.createDocumentFragment();

  rows.forEach((row, index) => {
    const speaker = speakerLabel(row);
    if (speaker) {
      const tag = document.createElement("span");
      tag.className = "speaker-inline";
      tag.textContent = speaker;
      tag.title = [
        `speaker-status=${row.speaker_status || "-"}`,
        `score=${row.speaker_last_score ?? "-"}`,
        `speaker-segments=${speakerSegments(row).length}`,
      ].join(" · ");
      frag.appendChild(tag);
    }

    const span = document.createElement("span");
    const st = segmentState(row);
    span.className = `seg ${row.__justCorrected ? "corrected" : st}`;
    span.dataset.id = row.id;
    span.title = [
      `turn=${row.turn_id || "-"}`,
      `segment=${row.segment_index ?? "-"}`,
      `speaker=${speaker || "-"}`,
      `source=${row.asr_source || "-"}`,
      `status=${row.verification_status || "-"}`,
      `revision=${row.revision ?? "-"}`,
    ].join(" · ");
    span.textContent = cleanText(row.text);
    frag.appendChild(span);

    if (index < rows.length - 1) {
      frag.appendChild(document.createTextNode(" "));
    }
  });

  ui.longTranscript.replaceChildren(frag);
  ui.longTranscript.scrollTop = ui.longTranscript.scrollHeight;
}

function renderLiveTail() {
  const p = state.currentPartial;
  if (!p || !cleanText(p.text)) {
    ui.liveTailWrap.classList.add("hidden");
    ui.liveTail.textContent = "";
    return;
  }
  ui.liveTail.textContent = cleanText(p.text);
  ui.liveTailWrap.classList.remove("hidden");
}

function translatedRows() {
  return sortedSegments().filter((r) => cleanText(r.translation));
}

function currentTranslationText() {
  return punctuatedJoin(translatedRows().map((r) => r.translation));
}

function renderTranslation() {
  if (!ui.showTranslation.checked) {
    ui.translationWrap.classList.add("hidden");
    return;
  }

  const sourceRows = sortedSegments();
  const rows = sourceRows.filter((row) => cleanText(row.translation));
  const h = state.health || {};

  const languageHint =
    rows.map((row) => row.translation_language).find(Boolean) || "";
  const target = String(languageHint).toLowerCase();
  const source =
    target === "en"
      ? "VI"
      : target === "vi"
        ? "EN"
        : ui.language.value.toUpperCase();
  const targetLabel =
    target
      ? target.toUpperCase()
      : (ui.language.value === "vi" ? "EN" : "VI");

  ui.translationLabel.textContent =
    `TRANSLATION · FINAL ONLY · ${source} → ${targetLabel}`;

  if (rows.length) {
    const frag = document.createDocumentFragment();

    rows.forEach((row, index) => {
      if (index > 0) frag.appendChild(document.createTextNode(" "));

      const span = document.createElement("span");
      span.className =
        `translation-seg final ${row.__translationJustCorrected ? "corrected" : ""}`;
      span.dataset.turn = String(row.turn_id || row.id || "");
      span.title = [
        "FINAL cumulative translation",
        `speaker=${speakerLabel(row) || "-"}`,
        `turn=${row.turn_id || row.id || "-"}`,
        `source-rev=${row.translation_source_revision ?? row.source_revision ?? "-"}`,
        `status=${row.translation_status || "-"}`,
      ].join(" · ");
      span.textContent = cleanText(row.translation);
      frag.appendChild(span);
    });

    ui.translationText.classList.remove("empty");
    ui.translationText.replaceChildren(frag);
    ui.translationText.scrollTop = ui.translationText.scrollHeight;
  } else {
    ui.translationText.classList.add("empty");
    const placeholder = document.createElement("span");
    placeholder.className = "placeholder";

    const hasFinalPending = sourceRows.some(
      (row) => row.is_final === true && row.translation_pending === true
    );

    if (h.translation_warming === true) {
      const family = String(h.translation_model_family || "opus-v20");
      placeholder.textContent =
        `Đang khởi động ${family}… VIT STT vẫn hoạt động độc lập.`;
    } else if (h.translation_enabled === false) {
      placeholder.textContent = "Translation đang bị tắt trên backend.";
    } else if (h.translation_available === false) {
      placeholder.textContent =
        "Translator V20 chưa sẵn sàng; canonical STT vẫn độc lập.";
    } else if (hasFinalPending || Number(h.translation_queue || 0) > 0) {
      placeholder.textContent =
        "Logical turn đã FINAL; đang tạo bản dịch authoritative.";
    } else {
      placeholder.textContent =
        "Chờ logical turn FINAL. V20 không hiển thị LIVE/progressive translation.";
    }

    ui.translationText.replaceChildren(placeholder);
  }

  const q = Number(h.translation_queue || 0);
  const ms = Number(h.translation_request_last_ms || 0);
  const family = String(h.translation_model_family || "opus-v20");
  const available = h.translation_available === true;

  const meta = [
    "FINAL only",
    family,
    available ? "ready" : "not-ready",
    `q ${q}`,
  ];
  if (ms > 0) meta.push(`${Math.round(ms)} ms`);

  ui.translationStatus.textContent = meta.join(" · ");
  ui.translationWrap.classList.remove("hidden");
}

function renderTurns() {
  const entries = [...state.turns.entries()]
    .map(([turnId]) => {
      const rows = sortedTurnRows(turnId);
      const start = Math.min(...rows.map((r) => Number(r.started_ms || Infinity)));
      return { turnId, rows, start };
    })
    .filter((x) => x.rows.length)
    .sort((a, b) => a.start - b.start);

  if (!entries.length) {
    ui.turnList.innerHTML = '<div class="empty-note">Chưa có dữ liệu.</div>';
    return;
  }

  ui.turnList.innerHTML = "";
  for (const { turnId, rows } of entries) {
    const box = document.createElement("div");
    box.className = "turn-row";

    const indices = rows.map((r) => r.segment_index ?? 0);
    const corrected = rows.filter(
      (r) => String(r.verification_status || "").includes("corrected")
    ).length;
    const verified = rows.filter((r) => segmentState(r) === "verified").length;
    const duration = rows.reduce((sum, r) => sum + Number(r.audio_ms || 0), 0);
    const speaker =
      rows.map((r) => speakerLabel(r)).filter(Boolean).at(-1) || "pending";
    const speakerSegCount = Math.max(
      0,
      ...rows.map((r) => speakerSegments(r).length)
    );

    const meta = document.createElement("div");
    meta.className = "turn-meta";
    meta.textContent =
      `${turnId.slice(0, 10)}… · ${rows.length} seg · index ${Math.min(...indices)}→${Math.max(...indices)} · ` +
      `${(duration / 1000).toFixed(1)}s audio · ${verified} verified · ${corrected} corrected · ` +
      `${speaker} · speaker-seg ${speakerSegCount}`;

    const text = document.createElement("div");
    text.className = "turn-text";
    const speakerPrefix = document.createElement("span");
    speakerPrefix.className = "turn-speaker";
    speakerPrefix.textContent = speaker === "pending" ? "" : `${speaker}: `;
    text.appendChild(speakerPrefix);
    text.appendChild(document.createTextNode(punctuatedJoin(rows.map((r) => r.text))));

    box.append(meta, text);

    const translations = rows.map((r) => cleanText(r.translation)).filter(Boolean);
    if (translations.length) {
      const translated = document.createElement("div");
      translated.className = "turn-translation";
      const target = rows.map((r) => r.translation_language).find(Boolean);
      translated.textContent =
        `${target ? String(target).toUpperCase() : "TR"} FINAL: ${punctuatedJoin(translations)}`;
      box.appendChild(translated);
    }

    ui.turnList.appendChild(box);
  }
}

function renderTable() {
  const rows = sortedSegments();
  if (!rows.length) {
    ui.segmentTable.innerHTML =
      '<tr><td colspan="11" class="empty-cell">Chưa có transcript event.</td></tr>';
    return;
  }

  ui.segmentTable.innerHTML = "";
  rows.forEach((row, idx) => {
    const tr = document.createElement("tr");
    const st = row.__justCorrected ? "corrected" : segmentState(row);
    const turn = String(row.turn_id || "-");
    const values = [
      String(idx + 1),
      turn === "-" ? "-" : turn.slice(0, 8),
      String(row.segment_index ?? "-"),
    ];

    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });

    const stateTd = document.createElement("td");
    const pill = document.createElement("span");
    pill.className = `state-pill ${st}`;
    pill.textContent = st;
    stateTd.appendChild(pill);
    tr.appendChild(stateTd);

    for (const value of [
      row.asr_source || "-",
      row.revision ?? "-",
      row.audio_ms ? `${(Number(row.audio_ms) / 1000).toFixed(1)}s` : "-",
      row.processing_ms != null ? `${row.processing_ms}ms` : "-",
    ]) {
      const td = document.createElement("td");
      td.textContent = String(value);
      tr.appendChild(td);
    }

    const speakerTd = document.createElement("td");
    speakerTd.className = "speaker-cell";
    speakerTd.textContent = speakerLabel(row) || "—";
    speakerTd.title = [
      `status=${row.speaker_status || "-"}`,
      `score=${row.speaker_last_score ?? "-"}`,
      `segments=${speakerSegments(row).length}`,
    ].join(" · ");
    tr.appendChild(speakerTd);

    const textTd = document.createElement("td");
    textTd.className = "text-cell";
    textTd.textContent = cleanText(row.text);
    textTd.title = row.verification_status || "";
    tr.appendChild(textTd);

    const translationTd = document.createElement("td");
    translationTd.className = "translation-cell";
    translationTd.textContent = cleanText(row.translation) || "—";
    translationTd.title = row.translation_status || "";
    tr.appendChild(translationTd);

    ui.segmentTable.appendChild(tr);
  });
}

function renderAll() {
  renderLongTranscript();
  renderLiveTail();
  renderTranslation();
  renderTurns();
  renderTable();
}

function clearView() {
  state.segments.clear();
  state.turns.clear();
  state.order.clear();
  state.nextOrder = 1;
  state.currentPartial = null;
  state.translationPreviews.clear();
  state.eventLines = [];
  ui.eventLog.textContent = "";
  renderAll();
}

function sendJson(payload) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return false;
  state.ws.send(JSON.stringify(payload));
  return true;
}

function joinRoom() {
  if (!state.connected) return;
  sendJson({
    type: "join",
    name: ui.speaker.value.trim() || "Guest",
    room: ui.room.value.trim() || "longform-test",
    language: ui.language.value,
  });
}

async function connect() {
  if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
    state.ws.close();
    return;
  }

  let url;
  try {
    url = wsUrl();
  } catch (err) {
    alert(err.message);
    return;
  }

  setBadge(ui.connectionBadge, "Connecting…", "warn");
  ui.connectBtn.disabled = true;

  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.addEventListener("open", () => {
    state.connected = true;
    ui.connectBtn.disabled = false;
    ui.connectBtn.textContent = "Disconnect";
    ui.micBtn.disabled = false;
    setBadge(ui.connectionBadge, "Connected", "good");
    logEvent("WS OPEN", { url });
    joinRoom();
    startHealthPolling();
  });

  ws.addEventListener("close", () => {
    state.connected = false;
    state.joined = false;
    ui.connectBtn.disabled = false;
    ui.connectBtn.textContent = "Connect";
    ui.micBtn.disabled = true;
    setBadge(ui.connectionBadge, "Disconnected", "bad");
    logEvent("WS CLOSE");
    stopHealthPolling();
    if (state.capturing) stopCapture().catch(console.error);
  });

  ws.addEventListener("error", () => {
    logEvent("WS ERROR");
    setBadge(ui.connectionBadge, "WebSocket error", "bad");
  });

  ws.addEventListener("message", (event) => {
    if (typeof event.data !== "string") return;
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }
    handleMessage(msg);
  });
}

function handleMessage(msg) {
  switch (msg.type) {
    case "hello":
      state.clientId = msg.client_id || "";
      if (msg.version) setBadge(ui.versionBadge, `AI Box ${msg.version}`, "neutral");
      logEvent("HELLO", { client: msg.client_id, version: msg.version, protocol: msg.capture_protocol });
      break;

    case "joined":
      state.joined = true;
      logEvent("JOINED", { room: msg.room, language: msg.language, version: msg.version });
      break;

    case "capture_ready":
      if (state.capturing) {
        state.captureReady = true;
        flushPendingPcm();
      }
      logEvent("CAPTURE READY", msg);
      break;

    case "transcript_partial": {
      state.currentPartial = { ...msg };
      renderLiveTail();
      break;
    }

    case "translation_partial":
      // V20 UX is deliberately FINAL-only for translation. Ignore any
      // unexpected legacy preview without affecting STT or FINAL rows.
      logEvent("TRANSLATION_PARTIAL_IGNORED_V20", {
        turn: msg.turn_id || msg.utterance_id || "",
        status: msg.translation_status || "",
      });
      break;

    case "transcript":
      upsertTranscript(msg, false);
      break;

    case "transcript_replace":
      upsertTranscript({ ...msg, id: msg.replace_id || msg.id }, true);
      break;

    case "processing":
    case "processing_done":
      logEvent(msg.type.toUpperCase(), {
        stage: msg.stage,
        utterance: msg.utterance_id,
        queue: msg.queue,
        unresolved: msg.unresolved,
      });
      break;

    case "room_status":
      // Health polling is canonical for counters; this event is still useful for immediate queue visibility.
      if (msg.stats && msg.stats.queue != null) ui.mQueue.textContent = String(msg.stats.queue);
      break;

    case "error":
      logEvent("SERVER ERROR", msg);
      if (
        state.capturing
        && ["stt-unavailable", "storage-pressure", "capture-mode-conflict"].includes(msg.code)
      ) {
        stopForAudioTransportFault(`server-${msg.code || "capture-error"}`);
      }
      break;

    default:
      break;
  }
}

async function healthOnce() {
  const url = `${backendBase()}/api/health`;
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const h = await r.json();

    state.health = h;

    if (h.version) {
      setBadge(
        ui.versionBadge,
        `AI Box ${h.version}`,
        h.ready === false ? "warn" : "neutral"
      );
    }

    ui.mQueue.textContent = String(h.queue ?? "—");
    ui.mDropped.textContent =
      h.dropped_audio_seconds == null
        ? "—"
        : `${Number(h.dropped_audio_seconds).toFixed(1)}s`;
    ui.mPreviewErrors.textContent = String(h.vit_preview_errors ?? "—");
    ui.mFinalErrors.textContent = String(h.vit_final_errors ?? "—");

    ui.mSpeakerResults.textContent = String(h.speaker_results ?? "—");
    ui.mSpeakerErrors.textContent = String(h.speaker_errors ?? "—");

    ui.mTranslationQueue.textContent = String(h.translation_queue ?? "—");
    ui.mTranslationDone.textContent = String(h.translation_completed ?? "—");
    ui.mTranslationErrors.textContent = String(h.translation_errors ?? "—");

    ui.mFinalDecoder.textContent = h.final_decoder
      ? `${h.final_decoder}${h.final_max_active_paths ? ` / ${h.final_max_active_paths}` : ""}`
      : "—";

    ui.mDropped.parentElement.classList.toggle(
      "metric-alert",
      Number(h.dropped_audio_seconds || 0) > 0
    );
    ui.mPreviewErrors.parentElement.classList.toggle(
      "metric-alert",
      Number(h.vit_preview_errors || 0) > 0
    );
    ui.mFinalErrors.parentElement.classList.toggle(
      "metric-alert",
      Number(h.vit_final_errors || 0) > 0
    );
    ui.mSpeakerErrors.parentElement.classList.toggle(
      "metric-warn",
      Number(h.speaker_errors || 0) > 0
    );
    ui.mTranslationErrors.parentElement.classList.toggle(
      "metric-warn",
      Number(h.translation_errors || 0) > 0
    );

    renderTranslation();
  } catch (err) {
    logEvent("HEALTH ERROR", { message: String(err.message || err) });
  }
}

function startHealthPolling() {
  stopHealthPolling();
  healthOnce();
  state.healthTimer = setInterval(healthOnce, 2000);
}

function stopHealthPolling() {
  if (state.healthTimer) clearInterval(state.healthTimer);
  state.healthTimer = null;
}


const MIC_DEVICE_STORAGE_KEY = "hlmeet.microphoneDeviceId";

function selectedMicrophoneLabel() {
  const option = ui.microphone?.selectedOptions?.[0];
  return option?.textContent?.trim() || "System default";
}

async function refreshMicrophones({ requestPermission = false } = {}) {
  if (!navigator.mediaDevices?.enumerateDevices) {
    throw new Error("This browser does not support microphone enumeration.");
  }

  let permissionStream = null;
  try {
    if (requestPermission) {
      permissionStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: false,
      });
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const microphones = devices.filter((device) => device.kind === "audioinput");

    const current = ui.microphone.value;
    const stored = localStorage.getItem(MIC_DEVICE_STORAGE_KEY) || "";
    const preferred = current || stored;

    ui.microphone.replaceChildren();
    ui.microphone.add(new Option("System default", ""));

    microphones.forEach((device, index) => {
      const label = device.label || `Microphone ${index + 1}`;
      ui.microphone.add(new Option(label, device.deviceId));
    });

    if (preferred && microphones.some((device) => device.deviceId === preferred)) {
      ui.microphone.value = preferred;
    } else {
      ui.microphone.value = "";
    }

    logEvent("MIC DEVICES", {
      count: microphones.length,
      selected: selectedMicrophoneLabel(),
      labelsVisible: microphones.some((device) => Boolean(device.label)),
    });
  } finally {
    permissionStream?.getTracks().forEach((track) => track.stop());
  }
}

async function startCapture() {
  if (!state.connected || state.capturing) return;

  joinRoom();

  // Freeze language for the lifetime of this capture. This prevents a UI
  // language change while getUserMedia/AudioContext are starting from
  // changing the audio-release policy mid-session.
  const captureLanguage = ui.language.value;

  const requestedDeviceId = ui.microphone.value;
  // STT evaluation profile: preserve the microphone waveform.
  // Browser DSP can suppress/reshape low-energy consonants, word onsets and
  // short utterances, so request raw mono capture for accuracy evaluation.
  const audioConstraints = {
    channelCount: { ideal: 1 },
    sampleRate: { ideal: 16000 },
    sampleSize: { ideal: 16 },
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
  };
  if (requestedDeviceId) {
    audioConstraints.deviceId = { exact: requestedDeviceId };
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: audioConstraints,
    video: false,
  });

  const track = stream.getAudioTracks()[0];
  const trackSettings = track?.getSettings?.() || {};
  const actualMicLabel = track?.label || selectedMicrophoneLabel();

  // Ask Web Audio to perform any hardware-rate conversion directly to
  // 16 kHz. Browser-native resampling is preferable to doing 48/44.1 -> 16 kHz
  // with a simple linear interpolator in JavaScript.
  let ctx;
  try {
    ctx = new AudioContext({
      latencyHint: "interactive",
      sampleRate: 16000,
    });
  } catch {
    // Compatibility fallback. pcm-worklet.js can still resample, and the
    // effective rate is logged below so an evaluation run is auditable.
    ctx = new AudioContext({ latencyHint: "interactive" });
  }
  // PHASE6G_VI_NATIVE_16K_FAIL_CLOSED
  //
  // Vietnamese accuracy release profile must not silently enter the worklet
  // linear-resampling compatibility path. English behavior is intentionally
  // unchanged.
  if (captureLanguage === "vi" && ctx.sampleRate !== 16000) {
    const actualRate = ctx.sampleRate;

    try {
      await ctx.close();
    } catch {}

    stream.getTracks().forEach((track) => track.stop());

    throw new Error(
      `Vietnamese STT accuracy mode requires a native 16000 Hz ` +
      `AudioContext; browser returned ${actualRate} Hz. ` +
      `Capture was stopped instead of using the linear-resampler fallback.`
    );
  }

  if (ctx.state === "suspended") await ctx.resume();
  await ctx.audioWorklet.addModule("./pcm-worklet.js");

  const source = ctx.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(ctx, "hl-pcm16-worklet", {
    processorOptions: {
      targetSampleRate: 16000,
      frameSamples: 320,
    },
  });

  // The node must remain connected for processing, but its audible output is muted.
  const silentGain = ctx.createGain();
  silentGain.gain.value = 0;
  source.connect(worklet);
  worklet.connect(silentGain);
  silentGain.connect(ctx.destination);

  worklet.port.onmessage = (event) => {
    const data = event.data;

    if (
      data &&
      typeof data === "object" &&
      !(data instanceof ArrayBuffer) &&
      data.type === "flush_ack"
    ) {
      const waiter = state.pcmFlushWaiter;
      state.pcmFlushWaiter = null;

      if (waiter) {
        waiter({
          acknowledged: true,
          samples: Number(data.samples || 0),
        });
      }

      return;
    }

    sendPcmFrame(data);
  };

  state.audioContext = ctx;
  state.mediaStream = stream;
  state.source = source;
  state.worklet = worklet;
  state.silentGain = silentGain;
  state.capturing = true;
  state.captureLanguage = captureLanguage;
  state.captureReady = false;
  state.audioTransportFault = false;
  clearPendingPcm();

  ui.microphone.disabled = true;
  ui.refreshMicsBtn.disabled = true;

  sendJson({ type: "capture_start", language: captureLanguage });
  ui.micBtn.textContent = "Stop microphone";
  ui.micBtn.classList.add("recording");
  setBadge(ui.captureBadge, "Automatic · AI Box VAD", "good");
  const dspActual = {
    echoCancellation: trackSettings.echoCancellation ?? null,
    noiseSuppression: trackSettings.noiseSuppression ?? null,
    autoGainControl: trackSettings.autoGainControl ?? null,
  };
  const native16k = ctx.sampleRate === 16000;

  ui.audioInfo.textContent =
    `${actualMicLabel} · input ${trackSettings.sampleRate || "?"} Hz · ` +
    `AudioContext ${ctx.sampleRate} Hz → PCM16 16 kHz · 20 ms frames · ` +
    `STT eval raw DSP requested OFF${native16k ? "" : " · resample fallback"}`;

  logEvent("MIC START", {
    profile: "stt-eval-raw-v1",
    microphone: actualMicLabel,
    selection: requestedDeviceId ? "explicit" : "system-default",
    trackSampleRate: trackSettings.sampleRate || null,
    trackChannelCount: trackSettings.channelCount || null,
    trackSampleSize: trackSettings.sampleSize || null,
    contextSampleRate: ctx.sampleRate,
    native16k,
    requestedDsp: {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
    actualDsp: dspActual,
    target: 16000,
    frameSamples: 320,
  });
}

async function flushVietnameseWorkletTail() {
  // VI-only release fix. English retains the exact pre-Phase6G stop behavior.
  if (
    state.captureLanguage !== "vi" ||
    !state.capturing ||
    !state.worklet
  ) {
    return {
      acknowledged: false,
      samples: 0,
      skipped: true,
    };
  }

  const result = await new Promise((resolve) => {
    let finished = false;

    const finish = (value) => {
      if (finished) return;
      finished = true;

      if (state.pcmFlushWaiter === finish) {
        state.pcmFlushWaiter = null;
      }

      resolve(value);
    };

    state.pcmFlushWaiter = finish;

    const timer = setTimeout(() => {
      finish({
        acknowledged: false,
        samples: 0,
        timeout: true,
      });
    }, 500);

    state.pcmFlushWaiter = (value) => {
      clearTimeout(timer);
      finish(value);
    };

    try {
      state.worklet.port.postMessage({
        type: "flush",
      });
    } catch (err) {
      clearTimeout(timer);
      finish({
        acknowledged: false,
        samples: 0,
        error: String(err?.message || err),
      });
    }
  });

  logEvent("VI PCM TAIL FLUSH", result);

  return result;
}


async function stopCapture() {
  if (!state.capturing && !state.audioContext && !state.mediaStream) return;

  // Flush exact sub-20ms browser tail before capture_stop. WebSocket preserves
  // message order, so the PCM binary message is queued before the JSON stop.
  // This path is deliberately VI-only.
  if (state.captureLanguage === "vi") {
    await flushVietnameseWorkletTail();
  }

  if (state.connected) sendJson({ type: "capture_stop" });
  state.capturing = false;
  state.captureReady = false;
  clearPendingPcm();

  try { state.source?.disconnect(); } catch {}
  try { state.worklet?.disconnect(); } catch {}
  try { state.silentGain?.disconnect(); } catch {}
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((t) => t.stop());
  }
  if (state.audioContext && state.audioContext.state !== "closed") {
    await state.audioContext.close();
  }

  state.audioContext = null;
  state.mediaStream = null;
  state.source = null;
  state.worklet = null;
  state.silentGain = null;
  state.captureLanguage = "";
  state.pcmFlushWaiter = null;

  ui.microphone.disabled = false;
  ui.refreshMicsBtn.disabled = false;

  ui.micBtn.textContent = "Start microphone";
  ui.micBtn.classList.remove("recording");
  setBadge(ui.captureBadge, "Mic off", "neutral");
  ui.audioInfo.textContent = "Audio: idle";
  state.currentPartial = null;
  renderLiveTail();
  logEvent("MIC STOP");
}

function currentTranscriptText() {
  return punctuatedJoin(sortedSegments().map((r) => r.text));
}

async function copyTranscript() {
  const text = currentTranscriptText();
  if (!text) return;
  await navigator.clipboard.writeText(text);
  const old = ui.copyBtn.textContent;
  ui.copyBtn.textContent = "Copied";
  setTimeout(() => { ui.copyBtn.textContent = old; }, 1200);
}

function exportTranscript() {
  const rows = sortedSegments();
  if (!rows.length) return;

  const lines = [
    "HL Meet V20 — VIT STT + anonymous speaker + FINAL translation",
    `Exported: ${new Date().toISOString()}`,
    `Room: ${ui.room.value.trim()}`,
    `Client name: ${ui.speaker.value.trim()}`,
    `Language: ${ui.language.value}`,
    "",
    "=== SOURCE TRANSCRIPT ===",
    currentTranscriptText(),
    "",
    "=== SPEAKER-AWARE TURNS ===",
  ];

  rows.forEach((r) => {
    const sid = speakerLabel(r) || "SPEAKER_PENDING";
    lines.push(`[${sid}] ${cleanText(r.text)}`);
    if (cleanText(r.translation)) {
      lines.push(
        `[${sid}] translation[${r.translation_language || "-"}] FINAL: ${cleanText(r.translation)}`
      );
    }
  });

  lines.push(
    "",
    "=== FINAL TRANSLATION ===",
    currentTranslationText() || "(no accepted FINAL translation)",
    "",
    "=== SEGMENTS ==="
  );

  rows.forEach((r, i) => {
    lines.push(
      `[${i + 1}] turn=${r.turn_id || "-"} seg=${r.segment_index ?? "-"} ` +
      `rev=${r.revision ?? "-"} source=${r.asr_source || "-"} ` +
      `state=${r.stability || "-"} status=${r.verification_status || "-"} ` +
      `speaker=${speakerLabel(r) || "-"} speaker_status=${r.speaker_status || "-"} ` +
      `speaker_score=${r.speaker_last_score ?? "-"} speaker_segments=${speakerSegments(r).length}`
    );
    lines.push(cleanText(r.text));
    if (cleanText(r.translation)) {
      lines.push(
        `translation[${r.translation_language || "-"}] status=${r.translation_status || "-"}:`
      );
      lines.push(cleanText(r.translation));
    }
    lines.push("");
  });

  const blob = new Blob([lines.join("\n")], {
    type: "text/plain;charset=utf-8",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `hl-meet-v20-${ui.room.value.trim() || "longform"}.txt`;
  document.body.appendChild(a);
  a.click();
  URL.revokeObjectURL(a.href);
  a.remove();
}

ui.connectBtn.addEventListener("click", () => connect().catch((err) => {
  logEvent("CONNECT EXCEPTION", { message: String(err) });
  ui.connectBtn.disabled = false;
}));

ui.micBtn.addEventListener("click", () => {
  const action = state.capturing ? stopCapture() : startCapture();
  action.catch((err) => {
    logEvent("MIC ERROR", { message: String(err.message || err) });
    alert(`Microphone error: ${err.message || err}`);
  });
});

ui.refreshMicsBtn.addEventListener("click", () => {
  refreshMicrophones({ requestPermission: true }).catch((err) => {
    logEvent("MIC ENUM ERROR", { message: String(err.message || err) });
    alert(`Microphone enumeration error: ${err.message || err}`);
  });
});

ui.microphone.addEventListener("change", () => {
  localStorage.setItem(MIC_DEVICE_STORAGE_KEY, ui.microphone.value || "");
  logEvent("MIC SELECT", {
    microphone: selectedMicrophoneLabel(),
    selection: ui.microphone.value ? "explicit" : "system-default",
  });
});

ui.clearBtn.addEventListener("click", clearView);
ui.copyBtn.addEventListener("click", () => copyTranscript().catch(console.error));
ui.exportBtn.addEventListener("click", exportTranscript);
ui.showTranslation.addEventListener("change", renderTranslation);

ui.language.addEventListener("change", () => {
  if (state.connected) {
    sendJson({ type: "language_pref", language: ui.language.value });
  }
});

window.addEventListener("beforeunload", () => {
  try {
    if (state.capturing) sendJson({ type: "capture_stop" });
    state.ws?.close();
  } catch {}
});

if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => {
    if (!state.capturing) {
      refreshMicrophones().catch((err) => {
        logEvent("MIC DEVICECHANGE ERROR", { message: String(err.message || err) });
      });
    }
  });
}

refreshMicrophones().catch((err) => {
  logEvent("MIC ENUM INIT ERROR", { message: String(err.message || err) });
});

renderAll();
