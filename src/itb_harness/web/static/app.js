const GRID_CELL_W = 68;
const GRID_CELL_H = 72;
const GRID_COLUMNS = 9;

const state = {
  runs: [],
  run: null,
  frameIndex: 0,
  showMarkers: true,
  showGrid: true,
  gridAnchors: new Map(),
  currentLevelMap: null,
  coordMode: "pixels",
  timelineRunId: null,
  timelineSignature: "",
  thumbRunId: null,
  openThinking: new Set(),
  live: null,
  liveSignature: "",
  liveTimer: null,
  liveSource: null,
  liveEnded: false,
  liveInitialized: false,
  liveActivityAt: 0,
  liveActivityAgeMs: 0,
  liveRefreshTimer: null,
  followLive: true,
  cursor: null,
};

const $ = (selector) => document.querySelector(selector);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

async function refreshRuns() {
  state.runs = await fetchJSON("/api/runs");
  renderRunSelect();
}

function frameName(index) {
  return state.run?.frames?.[index] ?? null;
}

function frameURL(name) {
  return `/api/runs/${encodeURIComponent(state.run.id)}/frames/${encodeURIComponent(name)}`;
}

function normalizeArgs(args) {
  if (typeof args !== "string") return args;
  try {
    return JSON.parse(args);
  } catch (error) {
    return args;
  }
}

function formatArgs(rawArgs) {
  const args = normalizeArgs(rawArgs);
  if (typeof args === "string") return args;
  if (!args || typeof args !== "object") return "";
  const point = args.point_2d;
  if (Array.isArray(point) && point.length === 2) {
    const suffix = args.with_details ? " +details" : "";
    return `(${point[0]}, ${point[1]})${suffix}`;
  }
  if (typeof args.steps !== "undefined") return `×${args.steps}`;
  if (typeof args.number !== "undefined") return `#${args.number}`;
  if (typeof args.text === "string") {
    const text = args.text.replace(/\s+/g, " ").trim();
    return text.length > 80 ? text.slice(0, 77) + "..." : text;
  }
  return "";
}

function usageText(usage) {
  if (!usage) return "";
  const parts = [];
  if (usage.prompt_tokens != null) parts.push(`${usage.prompt_tokens} in`);
  if (usage.cached_tokens != null) parts.push(`${usage.cached_tokens} cached`);
  if (usage.completion_tokens != null) parts.push(`${usage.completion_tokens} out`);
  return parts.join(" · ");
}

function renderMeta() {
  const meta = state.run?.metadata ?? {};
  const target = meta.target ?? {};
  const bits = [];
  if (target.label || target.title) bits.push(target.label || target.title);
  if (meta.model) bits.push(meta.model);
  if (meta.backend) bits.push(`capture ${meta.backend}`);
  if (meta.input) bits.push(`input ${meta.input}`);
  $("#meta").textContent = bits.join("  ·  ");
  $("#turn-count").textContent = `${state.run?.turns?.length ?? 0} turns · ${state.run?.frames?.length ?? 0} frames`;
}

function latestLevelMap(run) {
  const turns = run?.turns ?? [];
  for (let index = turns.length - 1; index >= 0; index -= 1) {
    if (Object.prototype.hasOwnProperty.call(turns[index], "level_map")) {
      return turns[index].level_map;
    }
  }
  return null;
}

function renderLevelMap(rows) {
  const panel = $("#level-map-panel");
  const node = $("#level-map");
  const meta = $("#level-map-meta");
  if (!panel || !node || !meta) return;

  if (!Array.isArray(rows) || !rows.length) {
    state.currentLevelMap = null;
    panel.hidden = true;
    node.textContent = "";
    meta.textContent = "";
    drawMarkers();
    return;
  }

  state.currentLevelMap = rows;
  const width = Array.isArray(rows[0]) ? rows[0].length : 0;
  node.textContent = rows
    .map((row) => (Array.isArray(row) ? row.map((cell) => `[${cell}]`).join(" ") : ""))
    .join("\n");
  meta.textContent = `${rows.length} × ${width}`;
  panel.hidden = false;
  drawMarkers();
}

function renderCurrentLevelMap() {
  const live = state.live;
  const viewingLive =
    state.followLive &&
    live?.active &&
    live.run_id &&
    state.run?.id === live.run_id;
  if (
    viewingLive &&
    Object.prototype.hasOwnProperty.call(live, "level_map")
  ) {
    renderLevelMap(live.level_map);
    return;
  }
  renderLevelMap(latestLevelMap(state.run));
}

function renderRunSelect() {
  const select = $("#run-select");
  select.replaceChildren();
  for (const run of state.runs) {
    const option = el(
      "option",
      null,
      `${run.id}  ·  ${run.model ?? "?"}  ·  ${run.frame_count} frames`
    );
    option.value = run.id;
    select.append(option);
  }
}

function actionClassName(action) {
  if (!action.ok) return "action error";
  if (action.tool === "click") return "action click";
  if (action.tool === "hover") return "action hover";
  if (action.tool === "inspect") return "action inspect";
  return "action";
}

function jumpTarget(action) {
  return action.frame_name || action.source_frame_name || null;
}

function renderTimeline(options = {}) {
  const timeline = $("#timeline");
  const runId = state.run?.id ?? null;
  const sameRun = state.timelineRunId === runId;
  const resetScroll = options.resetScroll ?? !sameRun;
  const toBottom = options.toBottom ?? false;
  const previousScrollTop = resetScroll
    ? 0
    : (options.scrollTop ?? timeline.scrollTop);
  const wasAtBottom =
    !resetScroll &&
    (options.wasAtBottom ??
      timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 40);

  if (!sameRun) state.openThinking.clear();
  timeline.replaceChildren();

  const turns = state.run?.turns ?? [];
  if (!turns.length) {
    timeline.append(el("div", "muted", "No turns recorded."));
    state.timelineRunId = runId;
    timeline.scrollTop = 0;
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const turn of turns) {
    const key = String(turn.step);
    const box = el("div", "turn");
    box.dataset.step = key;
    const head = el("div", "turn-head");
    head.append(el("span", null, `Turn ${turn.step}`));
    const usage = usageText(turn.usage);
    if (usage) head.append(el("span", "usage", usage));
    box.append(head);

    if (turn.guidance?.length) {
      const guidance = el("div", "guidance");
      guidance.append(el("div", "guidance-title", "interjection"));
      for (const text of turn.guidance) {
        guidance.append(el("div", "guidance-line", text));
      }
      box.append(guidance);
    }

    if (turn.thinking) {
      const details = el("details", "thinking");
      const ensureThinking = () => {
        if (!details.querySelector("pre")) {
          details.append(el("pre", null, turn.thinking));
        }
      };
      details.append(el("summary", null, "thinking"));
      details.open = state.openThinking.has(key);
      if (details.open) ensureThinking();
      details.addEventListener("toggle", () => {
        if (details.open) {
          state.openThinking.add(key);
          ensureThinking();
        } else {
          state.openThinking.delete(key);
          const pre = details.querySelector("pre");
          if (pre) pre.remove();
        }
      });
      box.append(details);
    }

    if (turn.content) {
      box.append(el("pre", "content", turn.content));
    }

    if (turn.actions?.length) {
      const actions = el("div", "actions");
      for (const action of turn.actions) {
        const target = jumpTarget(action);
        const button = el("button", actionClassName(action));
        if (target) {
          button.title = `Jump to ${target}`;
          button.addEventListener("click", () => jumpToFrame(target));
        } else {
          button.disabled = true;
        }

        const line = el("div", "action-line");
        line.append(el("span", `tool ${action.tool}`, action.tool));
        const args = formatArgs(action.arguments);
        if (args) {
          const argsNode = el("span", "action-args", args);
          argsNode.title = args;
          line.append(argsNode);
        }
        if (!action.ok) line.append(el("span", "badge", action.error || "error"));
        if (action.blocked === true) line.append(el("span", "badge blocked", "BLOCKED"));
        if (target) line.append(el("span", "frame-ref", target));
        button.append(line);

        const message = action.message || "";
        if (message) {
          button.append(
            el(
              "div",
              `action-result ${action.ok ? "ok" : "error"}`,
              message
            )
          );
        }
        actions.append(button);
      }
      box.append(actions);
    }

    fragment.append(box);
  }

  timeline.append(fragment);
  state.timelineRunId = runId;
  if (resetScroll) {
    timeline.scrollTop = toBottom ? timeline.scrollHeight : 0;
  } else if (wasAtBottom) {
    timeline.scrollTop = timeline.scrollHeight;
  } else {
    timeline.scrollTop = previousScrollTop;
  }
}

function renderThumbs() {
  const thumbs = $("#thumbs");
  const frames = state.run?.frames ?? [];
  const runId = state.run?.id ?? null;
  const sameRun = state.thumbRunId === runId;

  if (!sameRun || frames.length < thumbs.children.length) {
    thumbs.replaceChildren();
  }
  state.thumbRunId = runId;

  const fragment = document.createDocumentFragment();
  for (let index = thumbs.children.length; index < frames.length; index++) {
    const name = frames[index];
    const button = el("button", "thumb");
    button.title = name;
    const image = el("img");
    image.loading = "lazy";
    image.decoding = "async";
    image.src = frameURL(name);
    image.alt = name;
    button.append(image);
    button.addEventListener("click", () => showFrame(index));
    fragment.append(button);
  }
  if (fragment.childNodes.length) thumbs.append(fragment);

  for (let index = 0; index < thumbs.children.length; index++) {
    thumbs.children[index].classList.toggle(
      "active",
      index === state.frameIndex
    );
  }
}

function positionOverlay() {
  const image = $("#frame-img");
  const canvas = $("#overlay");
  const stage = $("#image-stage");
  if (!image.naturalWidth || !image.naturalHeight) return;

  const stageRect = stage.getBoundingClientRect();
  const imageRect = image.getBoundingClientRect();
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  canvas.style.left = `${imageRect.left - stageRect.left}px`;
  canvas.style.top = `${imageRect.top - stageRect.top}px`;
  canvas.style.width = `${imageRect.width}px`;
  canvas.style.height = `${imageRect.height}px`;
}

function drawLabel(ctx, text, x, y, color) {
  ctx.font = "600 18px ui-monospace, monospace";
  const paddingX = 7;
  const metrics = ctx.measureText(text);
  const width = metrics.width + paddingX * 2;
  const height = 26;
  let left = x + 14;
  let top = y - height - 8;
  if (left + width > ctx.canvas.width) left = x - width - 14;
  if (top < 0) top = y + 10;

  ctx.fillStyle = "rgba(10, 14, 18, 0.85)";
  ctx.fillRect(left, top, width, height);
  ctx.fillStyle = color;
  ctx.fillText(text, left + paddingX, top + 19);
}

function drawCrosshair(ctx) {
  if (!state.cursor) return;
  const { x, y } = state.cursor;
  ctx.save();
  ctx.strokeStyle = "rgba(255, 209, 102, 0.9)";
  ctx.lineWidth = 2;
  ctx.setLineDash([7, 7]);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(ctx.canvas.width, y);
  ctx.moveTo(x, 0);
  ctx.lineTo(x, ctx.canvas.height);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  const display = displayPoint(state.cursor);
  drawLabel(
    ctx,
    display ? `${display.x}, ${display.y}` : `${x}, ${y}`,
    x,
    y,
    "#ffd166"
  );
}

function isRedFloor(r, g, b) {
  // Covers the Helltaker floor reds around #8c1c31 and #700c29.
  return (
    r >= 80 &&
    r <= 200 &&
    g <= 80 &&
    b <= 110 &&
    r - g >= 45 &&
    r - b >= 30
  );
}

function gridAnchorKey() {
  return `${state.run?.id ?? ""}/${frameName(state.frameIndex) ?? ""}`;
}

function detectGridBottom(image) {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  if (!width || !height) return height;

  const sample = document.createElement("canvas");
  sample.width = width;
  sample.height = height;
  const ctx = sample.getContext("2d", { willReadFrequently: true });
  if (!ctx) return height;
  ctx.drawImage(image, 0, 0);

  let pixels;
  try {
    pixels = ctx.getImageData(0, 0, width, height).data;
  } catch (error) {
    return height;
  }

  // Scan upward for the first row containing at least one cell-width of
  // continuous floor red. That row is the bottom edge of the board.
  for (let y = height - 1; y >= 0; y -= 1) {
    let run = 0;
    let offset = y * width * 4;
    for (let x = 0; x < width; x += 1, offset += 4) {
      if (isRedFloor(pixels[offset], pixels[offset + 1], pixels[offset + 2])) {
        run += 1;
        if (run >= GRID_CELL_W) return y + 1;
      } else {
        run = 0;
      }
    }
  }

  return height;
}

function gridBottom() {
  const key = gridAnchorKey();
  if (state.gridAnchors.has(key)) return state.gridAnchors.get(key);

  const image = $("#frame-img");
  if (!image.naturalWidth || !image.naturalHeight) return null;
  const bottom = detectGridBottom(image);
  state.gridAnchors.set(key, bottom);
  return bottom;
}

function drawGrid(ctx) {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;
  const bottom = gridBottom();
  if (bottom == null) return;

  const left = width / 2 - (GRID_COLUMNS * GRID_CELL_W) / 2;
  const right = left + GRID_COLUMNS * GRID_CELL_W;

  ctx.save();
  ctx.strokeStyle = "rgba(120, 220, 255, 0.38)";
  ctx.lineWidth = 1;

  for (let col = 0; col <= GRID_COLUMNS; col += 1) {
    const x = Math.round(left + col * GRID_CELL_W) + 0.5;
    if (x < 0 || x > width) continue;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, Math.min(height, bottom));
    ctx.stroke();
  }

  // Anchor the bottom row to the first red floor row and propagate upward.
  // The top row is partial when the anchored height is not a multiple of 72.
  for (let y = bottom; y >= 0; y -= GRID_CELL_H) {
    const yy = Math.round(y) + 0.5;
    ctx.beginPath();
    ctx.moveTo(Math.max(0, left), yy);
    ctx.lineTo(Math.min(width, right), yy);
    ctx.stroke();
  }

  drawLevelMapAnnotations(ctx, width, height, bottom);

  ctx.restore();
}

function drawLevelMapAnnotations(ctx, width, height, bottom) {
  const rows = state.currentLevelMap;
  if (!Array.isArray(rows) || !rows.length) return;

  const columns = Math.max(
    ...rows.map((row) => (Array.isArray(row) ? row.length : 0))
  );
  if (!columns) return;

  const left = width / 2 - (columns * GRID_CELL_W) / 2;
  const fontSize = Math.max(12, Math.round(GRID_CELL_H * 0.3));
  const colors = {
    H: "#7ef0ff",
    S: "#f2f2f2",
    T: "#ff6b5f",
    D: "#ff8df0",
    R: "#ffb3c7",
    X: "#9aa7b4",
    K: "#ffd166",
    B: "#ff9f43",
  };

  ctx.save();
  ctx.font = `700 ${fontSize}px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";

  for (let row = 0; row < rows.length; row += 1) {
    const cells = Array.isArray(rows[row]) ? rows[row] : [];
    for (let col = 0; col < cells.length; col += 1) {
      const symbol = String(cells[col] ?? "").trim();
      if (!symbol) continue;

      const x = left + col * GRID_CELL_W + GRID_CELL_W / 2;
      const y =
        bottom - (rows.length - 1 - row) * GRID_CELL_H - GRID_CELL_H / 2;
      if (x < 0 || x > width || y < 0 || y > height) continue;

      if (symbol === ".") {
        ctx.fillStyle = "rgba(220, 230, 240, 0.5)";
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        continue;
      }

      const color = colors[symbol] ?? "#ffd166";
      const textWidth = ctx.measureText(symbol).width;
      ctx.fillStyle = "rgba(8, 12, 16, 0.72)";
      ctx.fillRect(
        x - textWidth / 2 - 5,
        y - fontSize / 2 - 3,
        textWidth + 10,
        fontSize + 6
      );
      ctx.fillStyle = color;
      ctx.fillText(symbol, x, y);
    }
  }

  ctx.restore();
}

function drawMarkers() {
  const canvas = $("#overlay");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.run) return;

  const name = frameName(state.frameIndex);
  if (!name) return;

  if (state.showGrid) drawGrid(ctx);

  if (state.showMarkers) {
    const radius = Math.max(10, canvas.width * 0.006);
    const half = radius + 4;

    for (const turn of state.run.turns ?? []) {
      for (const action of turn.actions ?? []) {
        if (!action.ok || !Array.isArray(action.point_2d)) continue;
        if (!(action.marker_frames ?? []).includes(name)) continue;

        const [x, y] = action.point_2d;
        const inspect = action.tool === "inspect";
        const hover = action.tool === "hover";
        const color = inspect
          ? "#54c8ff"
          : hover
            ? "#8fd3a8"
            : "#ff6259";

        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 3;

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x - half, y);
        ctx.lineTo(x + half, y);
        ctx.moveTo(x, y - half);
        ctx.lineTo(x, y + half);
        ctx.stroke();

        const label = inspect
          ? action.with_details
            ? "inspect +details"
            : "inspect"
          : hover
            ? "hover"
            : "click";
        drawLabel(ctx, label, x, y, color);
      }
    }
  }

  drawCrosshair(ctx);
}

function imagePoint(event) {
  const image = $("#frame-img");
  if (!image.naturalWidth || !image.naturalHeight) return null;
  const rect = image.getBoundingClientRect();
  if (
    event.clientX < rect.left ||
    event.clientX > rect.right ||
    event.clientY < rect.top ||
    event.clientY > rect.bottom
  ) {
    return null;
  }
  const x = Math.round(
    ((event.clientX - rect.left) / rect.width) * (image.naturalWidth - 1)
  );
  const y = Math.round(
    ((event.clientY - rect.top) / rect.height) * (image.naturalHeight - 1)
  );
  return {
    x: Math.max(0, Math.min(image.naturalWidth - 1, x)),
    y: Math.max(0, Math.min(image.naturalHeight - 1, y)),
  };
}

function displayPoint(point) {
  if (!point) return null;
  if (state.coordMode !== "normalized") return point;
  const image = $("#frame-img");
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  if (!width || !height) return point;
  return {
    x: Math.round((point.x / Math.max(1, width - 1)) * 1000),
    y: Math.round((point.y / Math.max(1, height - 1)) * 1000),
  };
}

function pointText(point) {
  const display = displayPoint(point);
  return display ? `x=${display.x}  y=${display.y}` : "x=— y=—";
}

function setCoordReadout(text) {
  $("#coords").textContent = text;
}

function updateCoordReadout() {
  setCoordReadout(pointText(state.cursor));
}

function clearCursor() {
  state.cursor = null;
  updateCoordReadout();
  drawMarkers();
}

function updateFrameToolbar() {
  const frames = state.run?.frames ?? [];
  $("#prev-frame").disabled = state.frameIndex <= 0;
  $("#next-frame").disabled = state.frameIndex >= frames.length - 1;
  const name = frameName(state.frameIndex);
  $("#frame-label").textContent = name
    ? `Frame ${state.frameIndex + 1}/${frames.length}  ·  ${name}`
    : "No frame";
}

function showFrame(index) {
  const frames = state.run?.frames ?? [];
  state.cursor = null;
  updateCoordReadout();

  if (!frames.length) {
    const image = $("#frame-img");
    image.dataset.frameName = "";
    image.style.display = "none";
    $("#overlay").style.display = "none";
    $("#empty").style.display = "block";
    state.frameIndex = 0;
    updateFrameToolbar();
    renderThumbs();
    return;
  }

  state.frameIndex = Math.max(0, Math.min(frames.length - 1, index));
  const name = frames[state.frameIndex];
  const image = $("#frame-img");
  const canvas = $("#overlay");

  $("#empty").style.display = "none";
  image.style.display = "block";
  canvas.style.display = "block";

  const sameImage =
    image.dataset.frameName === name &&
    image.complete &&
    image.naturalWidth > 0;
  if (sameImage) {
    positionOverlay();
    drawMarkers();
  } else {
    image.dataset.frameName = name;
    image.onload = () => {
      positionOverlay();
      drawMarkers();
    };
    image.src = frameURL(name);
  }

  updateFrameToolbar();
  renderThumbs();
}

function jumpToFrame(name) {
  const frames = state.run?.frames ?? [];
  const index = frames.indexOf(name);
  if (index >= 0) showFrame(index);
}

function timelineSignature(run) {
  return (run?.turns ?? [])
    .map((turn) => {
      const actions = turn.actions ?? [];
      const last = actions[actions.length - 1];
      return [
        turn.step,
        (turn.thinking ?? "").length,
        (turn.content ?? "").length,
        actions.length,
        (turn.guidance ?? []).join("\u0001"),
        last?.tool ?? "",
        last?.ok ? 1 : 0,
        (last?.message ?? "").length,
      ].join(":");
    })
    .join("|");
}

async function loadRun(runId, options = {}) {
  const { preserveFrame = false } = options;
  const previousFrame = frameName(state.frameIndex);
  const timeline = $("#timeline");
  const sameRun = state.run?.id === runId;
  const scrollTop = timeline.scrollTop;
  const wasAtBottom =
    timeline.scrollHeight - timeline.scrollTop - timeline.clientHeight < 40;
  if (!sameRun) {
    state.gridAnchors.clear();
    timeline.replaceChildren(el("div", "muted", "Loading…"));
  }
  const run = await fetchJSON(`/api/runs/${encodeURIComponent(runId)}`);
  state.run = run;

  const frames = run.frames ?? [];
  let index = Math.max(0, frames.length - 1);
  if (preserveFrame && previousFrame) {
    const previousIndex = frames.indexOf(previousFrame);
    if (previousIndex >= 0) index = previousIndex;
  }
  state.frameIndex = index;

  const select = $("#run-select");
  if ([...select.options].some((option) => option.value === runId)) {
    select.value = runId;
  }

  renderMeta();
  renderCurrentLevelMap();
  const signature = timelineSignature(run);
  if (!sameRun || signature !== state.timelineSignature) {
    renderTimeline({
      resetScroll: !sameRun,
      toBottom: !sameRun,
      scrollTop,
      wasAtBottom,
    });
    state.timelineSignature = signature;
  }
  showFrame(state.frameIndex);
}

function liveSignature(live) {
  if (!live?.active) return "";
  return [
    live.run_id,
    live.state,
    live.steps,
    live.actions,
    live.frames,
  ].join("|");
}

function updateLiveControls(live) {
  const panel = $("#live-panel");
  const stream = $("#live-stream");
  const steerForm = $("#live-steer-form");
  const steerInput = $("#live-steer-input");
  const guidance = $("#live-guidance");
  if (!live?.active) {
    panel.hidden = true;
    stream.hidden = true;
    steerForm.hidden = true;
    guidance.hidden = true;
    return;
  }
  panel.hidden = false;
  stream.hidden = false;
  const terminal = ["stopped", "stopping", "error"].includes(live.state);
  steerForm.hidden = terminal;
  steerInput.disabled = terminal;
  if (terminal) guidance.hidden = true;
  const label = live.busy ? `${live.state} (working)` : live.state;
  $("#live-state").textContent = label;
  const input = live.input ?? "unknown";
  const inputNode = $("#live-input");
  inputNode.textContent = input === "stub" ? "input: stub (not sent)" : `input: ${input}`;
  inputNode.className = input === "stub" ? "live-input warning" : "live-input";
  if (live.phase) $("#live-phase").textContent = live.phase;
  $("#live-error").textContent = live.error
    ? `error: ${live.error}`
    : live.reason
      ? `stopped: ${live.reason}`
      : "";

  const stateName = live.state;
  const pending = live.pending_guidance ?? 0;
  $("#live-steer-status").textContent = pending
    ? `${pending} queued`
    : "";
  $("#live-step").disabled = stateName !== "paused";
  $("#live-resume").disabled = stateName !== "paused";
  $("#live-pause").disabled = stateName !== "running";
  $("#live-stop").disabled = ["stopped", "stopping", "error"].includes(
    stateName
  );
  $("#live-reset").disabled =
    !!live.busy || !["paused", "stopped", "error"].includes(stateName);
}

function updateLiveActivity() {
  const live = state.live;
  const label = $("#live-activity");
  if (!live?.active) {
    label.textContent = "";
    return;
  }
  const phase = live.phase ?? state.livePhase ?? "idle";
  if (!["model_waiting", "thinking", "content"].includes(phase)) {
    label.textContent = "";
    label.className = "live-activity";
    return;
  }

  const ageMs = state.liveActivityAt
    ? Math.max(0, Date.now() - state.liveActivityAt)
    : state.liveActivityAgeMs || 0;
  const seconds = Math.floor(ageMs / 1000);
  let prefix = "last activity";
  if (phase === "model_waiting") prefix = "waiting for first token";
  else if (phase === "thinking") prefix = "thinking";
  else if (phase === "content") prefix = "streaming content";

  label.textContent = `${prefix} · ${seconds}s ago`;
  label.className = "live-activity";
  if (seconds >= 30) label.classList.add("stalled");
  else if (seconds >= 15) label.classList.add("warning");
}

function liveResetThinking() {
  state.liveInitialized = false;
  $("#live-thinking").textContent = "";
  $("#live-content").textContent = "";
}

function liveAppendThinking(text) {
  if (!text) return;
  state.liveInitialized = true;
  const node = $("#live-thinking");
  node.textContent += text;
  node.scrollTop = node.scrollHeight;
}

function liveAppendContent(text) {
  if (!text) return;
  state.liveInitialized = true;
  const node = $("#live-content");
  node.textContent += text;
  node.scrollTop = node.scrollHeight;
}

function requestLiveRunRefresh() {
  if (state.liveRefreshTimer) return;
  state.liveRefreshTimer = setTimeout(() => {
    state.liveRefreshTimer = null;
    pollLive();
  }, 120);
}

function liveIsTerminal(live) {
  return ["stopped", "stopping", "error"].includes(live?.state);
}

function appendGuidance(text, sent = false) {
  const box = $("#live-guidance");
  if (!text) return;
  box.hidden = false;
  const line = el(
    "div",
    `live-guidance-line${sent ? " sent" : ""}`,
    text
  );
  box.append(line);
  while (box.children.length > 30) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

function handleLiveEvent(event) {
  state.liveActivityAt = Date.now();
  state.liveActivityAgeMs = 0;
  const type = event.type;

  // Stop is asynchronous: the model request may still be finishing. Do not
  // keep appending its stream after the user has asked to stop.
  if (
    ["thinking", "content"].includes(type) &&
    liveIsTerminal(state.live)
  ) {
    return;
  }

  if (type === "status") {
    if (event.state) {
      state.live = { ...(state.live ?? {}), ...event, active: true };
      updateLiveControls(state.live);
    }
    if (event.phase) {
      state.livePhase = event.phase;
      $("#live-phase").textContent = event.phase;
    }
    if (event.last_activity_age_ms != null) {
      state.liveActivityAgeMs = event.last_activity_age_ms;
      state.liveActivityAt = Date.now() - event.last_activity_age_ms;
    }
    if (event.phase === "model_waiting") {
      liveResetThinking();
    }
    if (!state.liveInitialized && event.thinking != null) {
      liveAppendThinking(event.thinking);
    }
    if (!state.liveInitialized && event.content != null) {
      liveAppendContent(event.content);
    }
    updateLiveActivity();
    return;
  }

  if (type === "thinking") {
    liveAppendThinking(event.text);
    state.livePhase = "thinking";
    $("#live-phase").textContent = "thinking";
    updateLiveActivity();
    return;
  }

  if (type === "content") {
    liveAppendContent(event.text);
    state.livePhase = "content";
    $("#live-phase").textContent = "content";
    updateLiveActivity();
    return;
  }

  if (["frame", "turn", "tool_result"].includes(type)) {
    requestLiveRunRefresh();
    return;
  }

  if (type === "guidance_queued") {
    appendGuidance(`queued: ${event.text ?? ""}`);
    $("#live-steer-status").textContent = "queued";
    return;
  }

  if (type === "guidance") {
    appendGuidance(`sent: ${event.text ?? ""}`, true);
    $("#live-steer-status").textContent = "";
    requestLiveRunRefresh();
    return;
  }

  if (type === "reset") {
    state.live = {
      ...(state.live ?? {}),
      active: true,
      state: "paused",
      phase: "idle",
      run_id: event.run_id,
      steps: 0,
      actions: 0,
      frames: 0,
      level_map: event.level_map ?? null,
      error: null,
      reason: null,
    };
    state.liveEnded = false;
    state.livePhase = "idle";
    liveResetThinking();
    updateLiveControls(state.live);
    renderCurrentLevelMap();
    requestLiveRunRefresh();
    return;
  }

  if (type === "error") {
    state.live = {
      ...(state.live ?? {}),
      active: true,
      state: "error",
      error: event.error ?? "model error",
    };
    state.liveEnded = true;
    state.livePhase = "error";
    $("#live-phase").textContent = "error";
    updateLiveControls(state.live);
    if (state.liveSource) {
      state.liveSource.close();
      state.liveSource = null;
    }
    requestLiveRunRefresh();
    return;
  }

  if (type === "stopped") {
    state.live = {
      ...(state.live ?? {}),
      active: true,
      state: "stopped",
      reason: event.reason ?? state.live?.reason,
    };
    state.liveEnded = true;
    state.livePhase = "stopped";
    $("#live-phase").textContent = "stopped";
    updateLiveControls(state.live);
    if (state.liveSource) {
      state.liveSource.close();
      state.liveSource = null;
    }
    requestLiveRunRefresh();
  }
}

function startLiveStream() {
  if (
    state.liveSource ||
    state.liveEnded ||
    !state.live?.active ||
    liveIsTerminal(state.live)
  ) {
    return;
  }
  const source = new EventSource("/api/live/events");
  state.liveSource = source;
  source.onmessage = (message) => {
    try {
      handleLiveEvent(JSON.parse(message.data));
    } catch (error) {
      $("#live-error").textContent = `event parse error: ${error}`;
    }
  };
  source.onerror = () => {
    if (state.live?.active && state.live?.state !== "stopped") {
      $("#live-error").textContent = "live event stream disconnected; reconnecting…";
    }
  };
}

async function pollLive() {
  try {
    const live = await fetchJSON("/api/live");
    state.live = live;
    updateLiveControls(live);
    renderCurrentLevelMap();

    if (!live?.active) {
      if (state.liveSource) {
        state.liveSource.close();
        state.liveSource = null;
      }
      return;
    }
    state.liveEnded = liveIsTerminal(live);
    state.livePhase = live.phase ?? state.livePhase;
    if (live.last_activity_age_ms != null) {
      state.liveActivityAt = Date.now() - live.last_activity_age_ms;
    }
    if (!state.liveSource) startLiveStream();
    updateLiveActivity();
    const signature = liveSignature(live);
    const changed = signature !== state.liveSignature;
    state.liveSignature = signature;

    if (!state.followLive) return;

    if (live.run_id && state.run?.id !== live.run_id) {
      await loadRun(live.run_id);
      return;
    }

    // Reload the run files when there is new turn/action/frame data. State-only
    // changes (running → paused) are handled by the control panel.
    const dataChanged =
      live.steps !== state.lastLiveSteps ||
      live.actions !== state.lastLiveActions ||
      live.frames !== state.lastLiveFrames;
    if (dataChanged || changed && !state.run) {
      state.lastLiveSteps = live.steps;
      state.lastLiveActions = live.actions;
      state.lastLiveFrames = live.frames;
      await loadRun(live.run_id);
    }
  } catch (error) {
    $("#live-error").textContent = `live status failed: ${error}`;
  }
}

async function liveCommand(command) {
  try {
    if (command === "stop" && state.live) {
      state.live = { ...state.live, state: "stopping" };
      state.livePhase = "stopping";
      updateLiveControls(state.live);
    }
    await fetch(`/api/live/${command}`, { method: "POST" });
    if (command === "reset") {
      state.liveEnded = false;
      liveResetThinking();
      await refreshRuns();
    }
    await pollLive();
    setTimeout(pollLive, 250);
  } catch (error) {
    $("#live-error").textContent = `command failed: ${error}`;
  }
}

async function init() {
  try {
    const [runs, live] = await Promise.all([
      fetchJSON("/api/runs"),
      fetchJSON("/api/live"),
    ]);
    state.runs = runs;
    renderRunSelect();
    updateLiveControls(live);

    if (live?.active && live.run_id) {
      state.live = live;
      state.liveEnded = liveIsTerminal(live);
      state.followLive = true;
      $("#follow-live").checked = true;
      state.lastLiveSteps = live.steps;
      state.lastLiveActions = live.actions;
      state.lastLiveFrames = live.frames;
      state.livePhase = live.phase ?? "idle";
      state.liveActivityAt = Date.now() - (live.last_activity_age_ms ?? 0);
      await loadRun(live.run_id);
      startLiveStream();
    } else if (state.runs.length) {
      const select = $("#run-select");
      select.value = state.runs[0].id;
      await loadRun(state.runs[0].id);
    } else {
      $("#timeline").replaceChildren(
        el("div", "muted", "No runs found. Start an itb-agent run first.")
      );
    }

    state.liveTimer = setInterval(pollLive, 1200);
    setInterval(updateLiveActivity, 1000);
  } catch (error) {
    $("#timeline").replaceChildren(
      el("div", "action-result error", `Could not load runs: ${error}`)
    );
  }
}

$("#run-select").addEventListener("change", (event) => {
  if (state.live?.active) {
    const following = event.target.value === state.live.run_id;
    state.followLive = following;
    $("#follow-live").checked = following;
  }
  loadRun(event.target.value).catch((error) => {
    $("#timeline").replaceChildren(
      el("div", "action-result error", `Could not load run: ${error}`)
    );
  });
});

$("#follow-live").addEventListener("change", (event) => {
  state.followLive = event.target.checked;
  if (state.followLive && state.live?.active && state.live.run_id) {
    loadRun(state.live.run_id).catch((error) => {
      $("#live-error").textContent = `could not follow live run: ${error}`;
    });
  }
});

$("#live-steer-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = $("#live-steer-input");
  const status = $("#live-steer-status");
  const text = input.value.trim();
  if (!text) return;
  status.textContent = "sending…";
  try {
    const response = await fetch("/api/live/steer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.detail || "steering unavailable");
    }
    input.value = "";
    status.textContent = "queued";
  } catch (error) {
    status.textContent = `failed: ${error}`;
  }
});

$("#live-step").addEventListener("click", () => liveCommand("step"));
$("#live-resume").addEventListener("click", () => liveCommand("resume"));
$("#live-pause").addEventListener("click", () => liveCommand("pause"));
$("#live-stop").addEventListener("click", () => liveCommand("stop"));
$("#live-reset").addEventListener("click", () => liveCommand("reset"));

$("#prev-frame").addEventListener("click", () => showFrame(state.frameIndex - 1));
$("#next-frame").addEventListener("click", () => showFrame(state.frameIndex + 1));

$("#coord-normalized").addEventListener("change", (event) => {
  state.coordMode = event.target.checked ? "normalized" : "pixels";
  updateCoordReadout();
  drawMarkers();
});

$("#show-grid").addEventListener("change", (event) => {
  state.showGrid = event.target.checked;
  drawMarkers();
});

$("#show-markers").addEventListener("change", (event) => {
  state.showMarkers = event.target.checked;
  drawMarkers();
});

$("#image-stage").addEventListener("mousemove", (event) => {
  const point = imagePoint(event);
  if (!point) {
    if (state.cursor) clearCursor();
    return;
  }
  state.cursor = point;
  updateCoordReadout();
  drawMarkers();
});

$("#image-stage").addEventListener("mouseleave", () => {
  clearCursor();
});

$("#image-stage").addEventListener("click", async (event) => {
  const point = imagePoint(event);
  if (!point) return;
  const display = displayPoint(point);
  const text = `${display.x}, ${display.y}`;
  try {
    await navigator.clipboard.writeText(text);
    setCoordReadout(`copied ${text}`);
    setTimeout(updateCoordReadout, 900);
  } catch (error) {
    updateCoordReadout();
  }
});

window.addEventListener("resize", () => {
  positionOverlay();
  drawMarkers();
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
  if (event.key === "ArrowLeft") showFrame(state.frameIndex - 1);
  if (event.key === "ArrowRight") showFrame(state.frameIndex + 1);
});

init();
