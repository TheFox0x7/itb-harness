const state = {
  runs: [],
  run: null,
  frameIndex: 0,
  showMarkers: true,
  live: null,
  liveSignature: "",
  liveTimer: null,
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

function frameName(index) {
  return state.run?.frames?.[index] ?? null;
}

function frameURL(name) {
  return `/api/runs/${encodeURIComponent(state.run.id)}/frames/${encodeURIComponent(name)}`;
}

function formatArgs(args) {
  if (!args || typeof args !== "object") return "";
  const point = args.point_2d;
  if (Array.isArray(point) && point.length === 2) {
    const suffix = args.with_details ? " +details" : "";
    return `(${point[0]}, ${point[1]})${suffix}`;
  }
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
  if (action.tool === "inspect") return "action inspect";
  return "action";
}

function jumpTarget(action) {
  return action.frame_name || action.source_frame_name || null;
}

function renderTimeline() {
  const timeline = $("#timeline");
  timeline.replaceChildren();
  const turns = state.run?.turns ?? [];
  if (!turns.length) {
    timeline.append(el("div", "muted", "No turns recorded."));
    return;
  }

  for (const turn of turns) {
    const box = el("div", "turn");
    const head = el("div", "turn-head");
    head.append(el("span", null, `Turn ${turn.step}`));
    const usage = usageText(turn.usage);
    if (usage) head.append(el("span", "usage", usage));
    box.append(head);

    if (turn.thinking) {
      const details = el("details", "thinking");
      details.append(el("summary", null, "thinking"));
      details.append(el("pre", null, turn.thinking));
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
        if (args) line.append(el("span", "action-args", args));
        if (!action.ok) line.append(el("span", "badge", action.error || "error"));
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

    timeline.append(box);
  }
}

function renderThumbs() {
  const thumbs = $("#thumbs");
  thumbs.replaceChildren();
  const frames = state.run?.frames ?? [];
  frames.forEach((name, index) => {
    const button = el("button", "thumb");
    button.title = name;
    if (index === state.frameIndex) button.classList.add("active");
    const image = el("img");
    image.src = frameURL(name);
    image.alt = name;
    button.append(image);
    button.addEventListener("click", () => showFrame(index));
    thumbs.append(button);
  });
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
  drawLabel(ctx, `${x}, ${y}`, x, y, "#ffd166");
}

function drawMarkers() {
  const canvas = $("#overlay");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!state.run) return;

  const name = frameName(state.frameIndex);
  if (!name) return;

  if (state.showMarkers) {
    const radius = Math.max(10, canvas.width * 0.006);
    const half = radius + 4;

    for (const turn of state.run.turns ?? []) {
      for (const action of turn.actions ?? []) {
        if (!action.ok || !Array.isArray(action.point_2d)) continue;
        if (!(action.marker_frames ?? []).includes(name)) continue;

        const [x, y] = action.point_2d;
        const inspect = action.tool === "inspect";
        const color = inspect ? "#54c8ff" : "#ff6259";

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

function setCoordReadout(text) {
  $("#coords").textContent = text;
}

function clearCursor() {
  state.cursor = null;
  setCoordReadout("x=— y=—");
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
  setCoordReadout("x=— y=—");

  if (!frames.length) {
    $("#frame-img").style.display = "none";
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

  image.onload = () => {
    positionOverlay();
    drawMarkers();
  };
  image.src = frameURL(name);

  updateFrameToolbar();
  renderThumbs();
}

function jumpToFrame(name) {
  const frames = state.run?.frames ?? [];
  const index = frames.indexOf(name);
  if (index >= 0) showFrame(index);
}

async function loadRun(runId, options = {}) {
  const { preserveFrame = false } = options;
  const previousFrame = frameName(state.frameIndex);
  $("#timeline").replaceChildren(el("div", "muted", "Loading…"));
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
  renderTimeline();
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
  if (!live?.active) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const label = live.busy ? `${live.state} (working)` : live.state;
  $("#live-state").textContent = label;
  $("#live-error").textContent = live.error
    ? `error: ${live.error}`
    : live.reason
      ? `stopped: ${live.reason}`
      : "";

  const stateName = live.state;
  $("#live-step").disabled = stateName !== "paused";
  $("#live-resume").disabled = stateName !== "paused";
  $("#live-pause").disabled = stateName !== "running";
  $("#live-stop").disabled = ["stopped", "stopping", "error"].includes(
    stateName
  );
}

async function pollLive() {
  try {
    const live = await fetchJSON("/api/live");
    state.live = live;
    updateLiveControls(live);

    if (!live?.active) return;
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
    await fetch(`/api/live/${command}`, { method: "POST" });
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
      state.followLive = true;
      $("#follow-live").checked = true;
      state.lastLiveSteps = live.steps;
      state.lastLiveActions = live.actions;
      state.lastLiveFrames = live.frames;
      await loadRun(live.run_id);
    } else if (state.runs.length) {
      const select = $("#run-select");
      select.value = state.runs[0].id;
      await loadRun(state.runs[0].id);
    } else {
      $("#timeline").replaceChildren(
        el("div", "muted", "No runs found. Start an itb-agent run first.")
      );
    }

    state.liveTimer = setInterval(pollLive, 600);
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

$("#live-step").addEventListener("click", () => liveCommand("step"));
$("#live-resume").addEventListener("click", () => liveCommand("resume"));
$("#live-pause").addEventListener("click", () => liveCommand("pause"));
$("#live-stop").addEventListener("click", () => liveCommand("stop"));

$("#prev-frame").addEventListener("click", () => showFrame(state.frameIndex - 1));
$("#next-frame").addEventListener("click", () => showFrame(state.frameIndex + 1));

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
  setCoordReadout(`x=${point.x}  y=${point.y}`);
  drawMarkers();
});

$("#image-stage").addEventListener("mouseleave", () => {
  clearCursor();
});

$("#image-stage").addEventListener("click", async (event) => {
  const point = imagePoint(event);
  if (!point) return;
  try {
    await navigator.clipboard.writeText(`${point.x}, ${point.y}`);
    setCoordReadout(`copied ${point.x}, ${point.y}`);
    setTimeout(() => {
      if (state.cursor) {
        setCoordReadout(`x=${state.cursor.x}  y=${state.cursor.y}`);
      }
    }, 900);
  } catch (error) {
    setCoordReadout(`x=${point.x}  y=${point.y}`);
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
