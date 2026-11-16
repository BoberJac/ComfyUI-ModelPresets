import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// Model Presets node: JSON-backed preset switcher.
//
// Model loading lives in Python. Sampler settings and resolution are applied to
// real nodes by this extension, because sampler_name/scheduler are combo
// widgets and cannot be driven by links. Targets are stored as graph id + node
// id, so they work inside subgraphs as well as at the root.
// ---------------------------------------------------------------------------

const NODE_TYPE = "ModelPresetsNode";
const TARGETS_KEY = "modelPresetsTargets";
const AUTO_SAVE_KEY = "modelPresetsAutoSave";
const AUTO_SAVE_DELAY = 900;
const POLL_INTERVAL = 600;

const DEFAULT_PRESET = {
  source: "model",
  model: "",
  weight_dtype: "default",
  sage_attention: "auto",
  torch_compile: false,
  upscale_model: "",
  width: 1024,
  height: 1536,
  divide: false,
  round_to: 16,
  hires_width: 1536,
  hires_height: 2304,
  base: { steps: 25, cfg: 3.0, sampler: "euler", scheduler: "simple", denoise: 1.0 },
  hires: { enabled: false, steps: 15, cfg: 3.0, sampler: "euler_ancestral", scheduler: "normal", denoise: 0.45 },
};

const BASE_BUTTON = `
  height: 28px;
  border-radius: 5px;
  border: 1px solid rgba(255,255,255,0.12);
  color: #e5e7eb;
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
  cursor: pointer;
  pointer-events: auto;
  transition: background 0.16s ease, border-color 0.16s ease;
`;

// ---------------------------- API helpers ---------------------------------

async function fetchJson(url, options) {
  const response = await api.fetchApi(url, options);
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.error) detail = body.error;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return response.json();
}

const getPresets = () => fetchJson("/model_presets/get_presets");
const getOptions = () => fetchJson("/model_presets/options");

function savePreset(name, preset) {
  return fetchJson("/model_presets/save_preset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, preset }),
  });
}

function deletePreset(name) {
  return fetchJson("/model_presets/delete_preset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

// ---------------------------- graph helpers -------------------------------

function rootGraph() {
  return app.graph;
}

function allGraphs() {
  const root = rootGraph();
  if (!root) return [];
  const subgraphs = root.subgraphs instanceof Map ? [...root.subgraphs.values()] : [];
  return [root, ...subgraphs];
}

function findGraph(graphId) {
  return allGraphs().find((graph) => String(graph.id) === String(graphId)) ?? null;
}

function findNodeRef(ref) {
  if (!ref) return null;
  const graph = findGraph(ref.graphId);
  if (!graph) return null;
  return (graph._nodes ?? []).find((node) => String(node.id) === String(ref.nodeId)) ?? null;
}

function nodeRef(node) {
  return { graphId: String(node.graph?.id ?? rootGraph().id), nodeId: String(node.id) };
}

function nodeLabel(node) {
  const title = node.title || node.type || "node";
  return `${title} #${node.id}`;
}

function graphLabel(graph) {
  if (!graph) return "unknown";
  return String(graph.id) === String(rootGraph()?.id) ? "root" : graph.name || "subgraph";
}

function findWidget(node, names) {
  if (!node?.widgets) return null;
  return node.widgets.find((widget) => names.includes(widget.name)) ?? null;
}

function setWidget(node, names, value) {
  const widget = findWidget(node, names);
  if (!widget) return false;
  const before = widget.value;
  if (before === value) return true;
  widget.value = value;
  try {
    node.onWidgetChanged?.(widget.name, value, before, widget);
  } catch (error) {
    console.warn("ModelPresets: onWidgetChanged failed", error);
  }
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  node.graph?.incrementVersion?.();
  return true;
}

function applySamplerSettings(node, settings) {
  if (!node || !settings) return false;
  const steps = Number(settings.steps);
  const cfg = Number(settings.cfg);
  const denoise = Number(settings.denoise);
  let ok = true;
  if (Number.isFinite(steps)) ok = setWidget(node, ["steps"], steps) && ok;
  if (Number.isFinite(cfg)) ok = setWidget(node, ["cfg"], cfg) && ok;
  if (Number.isFinite(denoise)) ok = setWidget(node, ["denoise"], denoise) && ok;
  if (settings.sampler) ok = setWidget(node, ["sampler_name"], settings.sampler) && ok;
  if (settings.scheduler) ok = setWidget(node, ["scheduler"], settings.scheduler) && ok;
  return ok;
}

function followLinkedSizeSource(node, axis) {
  const widget = findWidget(node, [axis, axis === "width" ? "height" : "width"]);
  if (!widget) return node;
  const slot = node.inputs?.find((input) => input.widget?.name === widget.name && input.link != null);
  if (!slot) return node;
  const graph = node.graph;
  if (!graph) return null;
  const link = graph.links instanceof Map ? graph.links.get(slot.link) : graph.links?.[slot.link];
  if (!link) return null;
  const origin = (graph._nodes ?? []).find((n) => String(n.id) === String(link.origin_id));
  return origin ?? null;
}

function applySize(ref, value, axis) {
  let node = findNodeRef(ref);
  if (!node || !Number.isFinite(value)) return false;
  node = followLinkedSizeSource(node, axis);
  if (!node) {
    console.warn("ModelPresets: size widget is linked through a subgraph; pick the source node instead");
    return false;
  }
  const target = findWidget(node, [axis, axis === "width" ? "height" : "width"]);
  if (target) return setWidget(node, [target.name], value);
  return setWidget(node, ["value"], value);
}

function computeBaseSize(preset) {
  let width = Number(preset.width) || 0;
  let height = Number(preset.height) || 0;
  const step = Math.max(1, Number(preset.round_to) || 16);
  if (preset.divide) {
    width = Math.round(width / 2 / step) * step;
    height = Math.round(height / 2 / step) * step;
  } else if (Number(preset.round_to) > 1) {
    width = Math.round(width / step) * step;
    height = Math.round(height / step) * step;
  }
  return { width, height };
}

// ---------------------------- DOM helpers ---------------------------------

function make(tag, style, props = {}) {
  const element = document.createElement(tag);
  if (style) element.style.cssText = style;
  Object.assign(element, props);
  return element;
}

function labelled(text, control) {
  const wrap = make("div", "display:flex;flex-direction:column;gap:2px;min-width:0;");
  const label = make("div", "font-size:9px;color:#9ca3af;text-transform:uppercase;letter-spacing:.03em;");
  label.textContent = text;
  wrap.append(label, control);
  return wrap;
}

function selectInput(style = "") {
  return make(
    "select",
    `height:22px;border-radius:4px;border:1px solid rgba(255,255,255,0.12);background:#1b1f27;color:#e5e7eb;font-size:11px;padding:0 4px;min-width:0;${style}`
  );
}

function numberInput(step = 1) {
  return make(
    "input",
    "height:22px;border-radius:4px;border:1px solid rgba(255,255,255,0.12);background:#1b1f27;color:#e5e7eb;font-size:11px;padding:0 4px;min-width:0;box-sizing:border-box;",
    { type: "number", step: String(step) }
  );
}

function checkboxInput() {
  return make("input", "margin:0;", { type: "checkbox" });
}

function setSelectOptions(select, values, current) {
  const list = [...values];
  if (current !== undefined && current !== null && current !== "" && !list.includes(current)) {
    list.unshift(current);
  }
  select.replaceChildren(
    ...list.map((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      return option;
    })
  );
  if (current !== undefined && current !== null) select.value = current;
}

// ---------------------------- extension -----------------------------------

app.registerExtension({
  name: "ModelPresets.Panel",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      const node = this;

      node.setSize?.([560, 640]);

      const presetWidget = node.widgets?.find((widget) => widget.name === "preset");
      if (!presetWidget) {
        console.error("ModelPresets: preset widget not found");
        return result;
      }

      const state = {
        presets: {},
        options: {
          models: [],
          checkpoints: [],
          vaes: [],
          upscale_models: [],
          samplers: ["euler", "euler_ancestral", "dpmpp_2m"],
          schedulers: ["normal", "simple", "karras"],
          sage_modes: ["auto", "disabled"],
          weight_dtypes: ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],
        },
        loadedPreset: null,
        creatingNew: false,
        isLoading: false,
        dirty: false,
      };

      let autoSaveTimer = null;
      let pollId = null;
      let applyRetries = [];

      const container = make(
        "div",
        "display:flex;flex-direction:column;gap:6px;width:100%;box-sizing:border-box;padding:2px 0 0;pointer-events:auto;"
      );

      // ---- model / loader row ----
      const sourceSelect = selectInput();
      const modelSelect = selectInput();
      const dtypeSelect = selectInput();
      const sageSelect = selectInput();
      const compileCheckbox = checkboxInput();
      const upscaleSelect = selectInput();

      const rowLoader = make("div", "display:grid;grid-template-columns:90px 1fr;gap:4px;");
      rowLoader.append(
        labelled("source", sourceSelect),
        labelled("model", modelSelect),
        labelled("dtype", dtypeSelect),
        labelled("sage", sageSelect)
      );

      const rowUpscale = make("div", "display:flex;flex-direction:column;gap:2px;");
      rowUpscale.append(labelled("upscale model", upscaleSelect));

      const compileWrap = make("label", "display:flex;align-items:center;gap:6px;font-size:10px;color:#9ca3af;");
      compileWrap.append(compileCheckbox, document.createTextNode("torch.compile"));

      // ---- resolution row ----
      const widthInput = numberInput(16);
      const heightInput = numberInput(16);
      const divideCheckbox = checkboxInput();
      const roundInput = numberInput(16);
      const hiresWidthInput = numberInput(16);
      const hiresHeightInput = numberInput(16);

      const rowSize = make("div", "display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:4px;");
      rowSize.append(
        labelled("width", widthInput),
        labelled("height", heightInput),
        labelled("hires W", hiresWidthInput),
        labelled("hires H", hiresHeightInput)
      );

      const divideWrap = make("label", "display:flex;align-items:center;gap:6px;font-size:10px;color:#9ca3af;");
      divideWrap.append(divideCheckbox, document.createTextNode("base = target / 2"));
      const roundWrap = make("div", "display:flex;align-items:center;gap:6px;font-size:10px;color:#9ca3af;");
      const roundLabel = document.createTextNode("round to");
      roundWrap.append(roundLabel, roundInput);

      const rowSizeOptions = make("div", "display:flex;gap:14px;align-items:flex-end;");
      rowSizeOptions.append(divideWrap, roundWrap);

      // ---- sampler rows ----
      function samplerRow(prefix, isHires) {
        const enabled = checkboxInput();
        const steps = numberInput(1);
        const cfg = numberInput(0.1);
        const sampler = selectInput();
        const scheduler = selectInput();
        const denoise = numberInput(0.01);

        const row = make("div", "display:grid;grid-template-columns:1fr 1fr 1.2fr 1.2fr;gap:4px;");
        row.append(
          labelled("steps", steps),
          labelled("cfg", cfg),
          labelled("sampler", sampler),
          labelled("scheduler", scheduler)
        );
        const second = make("div", "display:flex;align-items:flex-end;gap:8px;");
        second.append(labelled("denoise", denoise));
        if (isHires) {
          const wrap = make("label", "display:flex;align-items:center;gap:6px;font-size:10px;color:#9ca3af;margin-left:auto;");
          wrap.append(enabled, document.createTextNode("enabled"));
          second.append(wrap);
        }

        const group = make(
          "div",
          "display:flex;flex-direction:column;gap:3px;border:1px solid rgba(255,255,255,0.08);border-radius:6px;padding:5px;"
        );
        const title = make("div", "font-size:10px;font-weight:700;color:#cbd5e1;");
        title.textContent = isHires ? "HIRES" : "BASE";
        group.append(title, row, second);
        return { group, enabled, steps, cfg, sampler, scheduler, denoise };
      }

      const baseRow = samplerRow("base", false);
      const hiresRow = samplerRow("hires", true);

      // ---- targets (canvas role binding) ----
      const targetStatus = make(
        "div",
        "font-size:10px;color:#9ca3af;white-space:pre-wrap;line-height:1.35;min-height:26px;"
      );

      function pickButton(role, text) {
        const button = make("button", `${BASE_BUTTON} flex:1;`, { type: "button" });
        button.textContent = text;
        button.onclick = () => armPick(role, button);
        return button;
      }

      // Pick mode survives graph navigation: press the button, walk into another
      // subgraph, click the target node. The node under the pointer is resolved
      // directly; the selection API is not reliable in the Vue renderer.
      let pendingPick = null;

      function eventToNode(event) {
        const canvas = app.canvas;
        const element = canvas?.canvas;
        if (!element || !element.contains?.(event.target)) return null;
        let pos = null;
        try {
          pos = canvas.convertEventToCanvasOffset?.(event) ?? null;
        } catch {
          pos = null;
        }
        if (!pos) return null;
        try {
          return canvas.graph?.getNodeOnPos?.(pos[0], pos[1], canvas.visible_nodes) ?? null;
        } catch {
          return null;
        }
      }

      function validateTarget(role, target) {
        if (role === "base" || role === "hires") {
          if (target.isSubgraphNode?.()) {
            return "That is a subgraph instance: open it and pick the KSampler inside.";
          }
          if (!findWidget(target, ["steps"]) || !findWidget(target, ["cfg"])) {
            return "That node has no steps/cfg widgets: pick a KSampler.";
          }
        } else if (role === "upscale") {
          if (!findWidget(target, ["model_name"])) {
            return "That node has no model_name widget: pick an UpscaleModelLoader.";
          }
        } else if (!findWidget(target, ["width", "height", "value"])) {
          return "That node has no width/height/value widget.";
        }
        return null;
      }

      function bindTarget(role, target) {
        const problem = validateTarget(role, target);
        if (problem) {
          setStatus(problem, "warn");
          return false;
        }
        node.properties = node.properties ?? {};
        node.properties[TARGETS_KEY] = node.properties[TARGETS_KEY] ?? {};
        node.properties[TARGETS_KEY][role] = nodeRef(target);
        refreshTargets();
        applyCurrent(`bound ${role}: ${nodeLabel(target)}`);
        return true;
      }

      function onPickPointerDown(event) {
        if (!pendingPick) return;
        if (!app.canvas?.canvas?.contains?.(event.target)) return;
        const target = eventToNode(event);
        if (!target) {
          setStatus("No node under the cursor. Click directly on the node body.", "warn");
          return;
        }
        const { role, button, label } = pendingPick;

        // A sampler lives inside a subgraph: open it and keep pick mode armed.
        if (target.isSubgraphNode?.() && (role === "base" || role === "hires")) {
          event.preventDefault();
          event.stopPropagation();
          const name = target.subgraph?.name ?? "subgraph";
          try {
            app.canvas.openSubgraph(target.subgraph, target);
            setStatus(`Opened "${name}": click the KSampler inside.`);
          } catch (error) {
            console.warn("ModelPresets: could not open subgraph", error);
            setStatus(`Could not open "${name}"; use the subgraph enter button.`, "warn");
          }
          return;
        }

        event.preventDefault();
        event.stopPropagation();
        pendingPick = null;
        button.textContent = label;
        if (!bindTarget(role, target)) {
          // Keep pick mode armed so the next click can fix the mistake.
          pendingPick = { role, button, label };
          button.textContent = "click a node...";
        }
      }

      function armPick(role, button) {
        if (pendingPick?.role === role) {
          pendingPick.button.textContent = pendingPick.label;
          pendingPick = null;
          return;
        }
        if (pendingPick) pendingPick.button.textContent = pendingPick.label;
        pendingPick = { role, button, label: button.textContent };
        button.textContent = "click a node...";
        setStatus("Pick mode: click the target node. A subgraph instance opens itself for sampler roles.");
      }

      const targetsGrid = make("div", "display:grid;grid-template-columns:repeat(4,1fr);gap:4px;");
      targetsGrid.append(
        pickButton("base", "Pick base"),
        pickButton("hires", "Pick hires"),
        pickButton("width", "Pick W"),
        pickButton("height", "Pick H"),
        pickButton("hiresWidth", "Pick hiW"),
        pickButton("hiresHeight", "Pick hiH"),
        pickButton("upscale", "Pick upscale")
      );

      // ---- main buttons ----
      const autoSaveButton = make("button", `${BASE_BUTTON} flex:1;`, { type: "button" });
      const saveButton = make("button", `${BASE_BUTTON} flex:1;`, { type: "button" });
      const newButton = make("button", `${BASE_BUTTON} flex:1;`, { type: "button" });
      const deleteButton = make("button", `${BASE_BUTTON} flex:1;`, { type: "button" });
      const importButton = make("button", `${BASE_BUTTON} flex:1;`, { type: "button" });
      const nameInput = make(
        "input",
        "height:24px;border-radius:5px;border:1px solid rgba(255,255,255,0.12);background:#1b1f27;color:#e5e7eb;font-size:11px;padding:0 6px;flex:1;min-width:0;",
        { type: "text", placeholder: "Preset name" }
      );

      const rowName = make("div", "display:flex;gap:4px;align-items:center;");
      rowName.append(nameInput, saveButton, newButton, deleteButton);

      const rowButtons = make("div", "display:grid;grid-template-columns:1.2fr 1fr;gap:4px;");
      rowButtons.append(autoSaveButton, importButton);

      container.append(
        rowLoader,
        rowUpscale,
        compileWrap,
        rowSize,
        rowSizeOptions,
        baseRow.group,
        hiresRow.group,
        rowName,
        rowButtons,
        targetsGrid,
        targetStatus
      );

      saveButton.textContent = "Save Preset";
      newButton.textContent = "New Preset";
      deleteButton.textContent = "Delete Preset";
      importButton.textContent = "Import from workflow";

      // ---- behaviour ----
      function isAutoSaveEnabled() {
        node.properties = node.properties ?? {};
        return node.properties[AUTO_SAVE_KEY] !== false;
      }

      function refreshAutoSaveButton() {
        const on = isAutoSaveEnabled();
        autoSaveButton.textContent = on ? "Auto-save: On" : "Auto-save: Off";
        autoSaveButton.style.background = on ? "rgba(34,197,94,0.15)" : "rgba(255,255,255,0.045)";
        autoSaveButton.style.borderColor = on ? "rgba(34,197,94,0.3)" : "rgba(255,255,255,0.10)";
      }

      function setStatus(text, tone = "info") {
        targetStatus.textContent = text;
        targetStatus.style.color =
          tone === "error" ? "#f87171" : tone === "warn" ? "#fbbf24" : "#9ca3af";
      }

      function updateSamplerSelects() {
        for (const row of [baseRow, hiresRow]) {
          setSelectOptions(row.sampler, state.options.samplers, row.sampler.value);
          setSelectOptions(row.scheduler, state.options.schedulers, row.scheduler.value);
        }
      }

      function updateModelSelect() {
        const source = sourceSelect.value;
        const values = source === "checkpoint" ? state.options.checkpoints : state.options.models;
        setSelectOptions(modelSelect, values, modelSelect.value);
      }

      function setUpscaleOptions(current) {
        const values = ["", ...(state.options.upscale_models || [])];
        if (current && !values.includes(current)) values.push(current);
        upscaleSelect.replaceChildren(
          ...values.map((value) => {
            const option = document.createElement("option");
            option.value = value;
            option.textContent = value || "(leave unchanged)";
            return option;
          })
        );
        upscaleSelect.value = current || "";
      }

      function fillPanel(preset) {
        state.isLoading = true;
        try {
          sourceSelect.value = preset.source ?? "model";
          const sourceValues = sourceSelect.value === "checkpoint" ? state.options.checkpoints : state.options.models;
          setSelectOptions(modelSelect, sourceValues, preset.model ?? "");
          setSelectOptions(dtypeSelect, state.options.weight_dtypes, preset.weight_dtype ?? "default");
          setSelectOptions(sageSelect, state.options.sage_modes, preset.sage_attention ?? "auto");
          setUpscaleOptions(preset.upscale_model ?? "");
          compileCheckbox.checked = Boolean(preset.torch_compile);
          widthInput.value = preset.width ?? 1024;
          heightInput.value = preset.height ?? 1536;
          divideCheckbox.checked = Boolean(preset.divide);
          roundInput.value = preset.round_to ?? 16;
          hiresWidthInput.value = preset.hires_width ?? "";
          hiresHeightInput.value = preset.hires_height ?? "";
          const base = preset.base ?? {};
          baseRow.steps.value = base.steps ?? 25;
          baseRow.cfg.value = base.cfg ?? 3;
          baseRow.denoise.value = base.denoise ?? 1;
          setSelectOptions(baseRow.sampler, state.options.samplers, base.sampler ?? "euler");
          setSelectOptions(baseRow.scheduler, state.options.schedulers, base.scheduler ?? "simple");
          const hires = preset.hires ?? {};
          hiresRow.enabled.checked = Boolean(hires.enabled);
          hiresRow.steps.value = hires.steps ?? 15;
          hiresRow.cfg.value = hires.cfg ?? 3;
          hiresRow.denoise.value = hires.denoise ?? 0.45;
          setSelectOptions(hiresRow.sampler, state.options.samplers, hires.sampler ?? "euler_ancestral");
          setSelectOptions(hiresRow.scheduler, state.options.schedulers, hires.scheduler ?? "normal");
        } finally {
          state.isLoading = false;
        }
      }

      function readPanel() {
        return {
          source: sourceSelect.value,
          model: modelSelect.value,
          weight_dtype: dtypeSelect.value,
          sage_attention: sageSelect.value,
          torch_compile: compileCheckbox.checked,
          upscale_model: upscaleSelect.value,
          width: parseInt(widthInput.value, 10) || 1024,
          height: parseInt(heightInput.value, 10) || 1536,
          divide: divideCheckbox.checked,
          round_to: parseInt(roundInput.value, 10) || 16,
          hires_width: parseInt(hiresWidthInput.value, 10) || 0,
          hires_height: parseInt(hiresHeightInput.value, 10) || 0,
          base: {
            steps: parseInt(baseRow.steps.value, 10) || 25,
            cfg: parseFloat(baseRow.cfg.value) || 3,
            sampler: baseRow.sampler.value,
            scheduler: baseRow.scheduler.value,
            denoise: parseFloat(baseRow.denoise.value) || 1,
          },
          hires: {
            enabled: hiresRow.enabled.checked,
            steps: parseInt(hiresRow.steps.value, 10) || 15,
            cfg: parseFloat(hiresRow.cfg.value) || 3,
            sampler: hiresRow.sampler.value,
            scheduler: hiresRow.scheduler.value,
            denoise: parseFloat(hiresRow.denoise.value) || 0.45,
          },
        };
      }

      function currentTargets() {
        node.properties = node.properties ?? {};
        return node.properties[TARGETS_KEY] ?? {};
      }

      function refreshTargets() {
        const targets = currentTargets();
        const lines = [];
        let lost = false;
        const describe = (role, label) => {
          const ref = targets[role];
          const target = findNodeRef(ref);
          if (!ref) return `${label}: not set`;
          if (!target) {
            lost = true;
            return `${label}: missing (${String(ref.graphId).slice(0, 8)}... #${ref.nodeId})`;
          }
          return `${label}: ${nodeLabel(target)} [${graphLabel(target.graph)}]`;
        };
        lines.push(describe("base", "base"));
        lines.push(describe("hires", "hires"));
        lines.push([describe("width", "W"), describe("height", "H")].join("  /  "));
        lines.push([describe("hiresWidth", "hiW"), describe("hiresHeight", "hiH")].join("  /  "));
        lines.push(describe("upscale", "upscale"));
        setStatus(lines.join("\n"), lost ? "error" : "info");
      }

      function applyCurrent(reason = "", { silent = false } = {}) {
        const preset = state.creatingNew ? readPanel() : state.presets[presetWidget.value] ?? readPanel();
        const targets = currentTargets();

        const baseNode = findNodeRef(targets.base);
        const hiresNode = findNodeRef(targets.hires);
        const sampledBase = applySamplerSettings(baseNode, preset.base);
        const sampledHires = preset.hires?.enabled ? applySamplerSettings(hiresNode, preset.hires) : true;

        const baseSize = computeBaseSize(preset);
        const sizedWidth = applySize(targets.width, baseSize.width, "width");
        const sizedHeight = applySize(targets.height, baseSize.height, "height");
        const hiresW = Number(preset.hires_width) || 0;
        const hiresH = Number(preset.hires_height) || 0;
        let sizedHiresWidth = true;
        let sizedHiresHeight = true;
        if (preset.hires?.enabled && hiresW) sizedHiresWidth = applySize(targets.hiresWidth, hiresW, "width");
        if (preset.hires?.enabled && hiresH) sizedHiresHeight = applySize(targets.hiresHeight, hiresH, "height");

        const upscaleRequested = Boolean(preset.upscale_model);
        const upscaleNode = findNodeRef(targets.upscale);
        const upscaleOk = !upscaleRequested || setWidget(upscaleNode, ["model_name"], preset.upscale_model);

        const missing = [];
        if (!sampledBase) missing.push("base sampler");
        if (!sizedWidth) missing.push("width target");
        if (!sizedHeight) missing.push("height target");
        if (preset.hires?.enabled && !sampledHires) missing.push("hires sampler");
        if (preset.hires?.enabled && hiresW && !sizedHiresWidth) missing.push("hires width");
        if (preset.hires?.enabled && hiresH && !sizedHiresHeight) missing.push("hires height");
        if (upscaleRequested && !upscaleOk) missing.push("upscale model target");
        if (!silent) {
          refreshTargets();
          if (missing.length) {
            setStatus(`Applied ${reason || "preset"}. Not set: ${missing.join(", ")}.`, "warn");
          } else {
            setStatus(`Applied ${reason || "preset"}.`);
          }
        }
        return missing.length === 0;
      }

      function applyWithRetry(reason, attempts = 4) {
        applyRetries.forEach(clearTimeout);
        applyRetries = [];
        if (applyCurrent(reason)) return;
        for (let i = 1; i <= attempts; i++) {
          applyRetries.push(
            setTimeout(() => {
              if (applyCurrent(reason, { silent: true })) {
                applyRetries.forEach(clearTimeout);
                applyRetries = [];
                applyCurrent(reason);
              }
            }, i * 250)
          );
        }
      }

      async function loadPreset(name, { apply = true } = {}) {
        const preset = state.presets[name];
        if (!preset) {
          console.warn(`ModelPresets: preset "${name}" not found`);
          return;
        }
        state.loadedPreset = name;
        fillPanel(preset);
        nameInput.value = name;
        if (apply) applyWithRetry(`preset "${name}"`);
      }

      function scheduleAutoSave() {
        state.dirty = true;
        clearTimeout(autoSaveTimer);
        autoSaveTimer = setTimeout(doAutoSave, AUTO_SAVE_DELAY);
      }

      async function doAutoSave() {
        autoSaveTimer = null;
        if (state.isLoading || state.creatingNew) return;
        if (!isAutoSaveEnabled()) return;
        const name = presetWidget.value;
        if (!name) return;
        const preset = readPanel();
        state.dirty = false;
        try {
          await savePreset(name, preset);
          state.presets[name] = preset;
          applyCurrent(`auto-save "${name}"`);
        } catch (error) {
          console.error("ModelPresets: auto-save failed", error);
          setStatus(`Auto-save failed: ${error.message}`, "error");
        }
      }

      async function flushBeforeQueue() {
        clearTimeout(autoSaveTimer);
        autoSaveTimer = null;
        if (!state.dirty || state.creatingNew || !isAutoSaveEnabled()) {
          applyCurrent("queue");
          return;
        }
        await doAutoSave();
      }

      async function refreshPresetList(preferred) {
        state.presets = await getPresets();
        const names = Object.keys(state.presets);
        if (!names.length) names.push("Default");
        if (presetWidget.options) presetWidget.options.values = names;
        if (preferred && names.includes(preferred)) presetWidget.value = preferred;
        else if (!names.includes(presetWidget.value)) presetWidget.value = names[0];
      }

      function bindInputs() {
        const inputs = [
          modelSelect, dtypeSelect, sageSelect, upscaleSelect, widthInput, heightInput, roundInput,
          hiresWidthInput, hiresHeightInput,
          baseRow.steps, baseRow.cfg, baseRow.sampler, baseRow.scheduler, baseRow.denoise,
          hiresRow.steps, hiresRow.cfg, hiresRow.sampler, hiresRow.scheduler, hiresRow.denoise,
        ];
        for (const input of inputs) {
          input.addEventListener("input", scheduleAutoSave);
          input.addEventListener("change", scheduleAutoSave);
        }
        for (const checkbox of [compileCheckbox, divideCheckbox, hiresRow.enabled]) {
          checkbox.addEventListener("change", scheduleAutoSave);
        }
        sourceSelect.addEventListener("change", () => {
          updateModelSelect();
          scheduleAutoSave();
        });
      }

      function bindButtons() {
        autoSaveButton.onclick = () => {
          node.properties = node.properties ?? {};
          node.properties[AUTO_SAVE_KEY] = !isAutoSaveEnabled();
          refreshAutoSaveButton();
          if (isAutoSaveEnabled()) scheduleAutoSave();
        };

        saveButton.onclick = async () => {
          const name = nameInput.value.trim();
          if (!name) {
            setStatus("Enter a preset name first.", "error");
            return;
          }
          const preset = readPanel();
          try {
            await savePreset(name, preset);
            state.creatingNew = false;
            state.dirty = false;
            await refreshPresetList(name);
            state.presets[name] = preset;
            presetWidget.value = name;
            state.loadedPreset = name;
            applyCurrent(`saved "${name}"`);
            setStatus(`Saved "${name}".`);
          } catch (error) {
            setStatus(`Save failed: ${error.message}`, "error");
          }
        };

        newButton.onclick = () => {
          state.creatingNew = true;
          state.loadedPreset = null;
          fillPanel({ ...DEFAULT_PRESET, model: modelSelect.value || state.options.models[0] || "" });
          nameInput.value = "My Model";
          nameInput.focus();
          setStatus("New preset: fill the fields and press Save Preset.");
        };

        deleteButton.onclick = async () => {
          const name = presetWidget.value;
          if (!name) return;
          if (!confirm(`Delete preset "${name}"?`)) return;
          try {
            await deletePreset(name);
            delete state.presets[name];
            const names = Object.keys(state.presets);
            if (!names.length) names.push("Default");
            if (presetWidget.options) presetWidget.options.values = names;
            presetWidget.value = names[0];
            state.creatingNew = false;
            await loadPreset(names[0]);
          } catch (error) {
            setStatus(`Delete failed: ${error.message}`, "error");
          }
        };

        importButton.onclick = () => importFromWorkflow();
      }

      // ---- import -----------------------------------------------------------
      function loaderModelName(loader) {
        const widget = findWidget(loader, ["model_name", "unet_name", "ckpt_name"]);
        return widget ? String(widget.value) : "";
      }

      function isLoader(node) {
        return ["AnimaBoosterLoader", "AnimaBoosterCheckpointLoader", "UNETLoader", "CheckpointLoaderSimple"].includes(node.type);
      }

      function samplerModelLink(graph, sampler) {
        const input = sampler.inputs?.find((slot) => slot.type === "MODEL" || slot.name === "model");
        return input?.link ?? null;
      }

      function getLink(graph, id) {
        if (id === null || id === undefined) return null;
        if (graph.links instanceof Map) return graph.links.get(id) ?? null;
        return graph.links?.[id] ?? null;
      }

      function samplersForLoader(graph, loader) {
        const samplers = (graph._nodes ?? []).filter((n) => n.type === "KSampler");
        const byModel = samplers.filter((sampler) => {
          const seen = new Set();
          let link = getLink(graph, samplerModelLink(graph, sampler));
          for (let depth = 0; link && depth < 8; depth++) {
            if (seen.has(link.id)) break;
            seen.add(link.id);
            const origin = (graph._nodes ?? []).find((n) => String(n.id) === String(link.origin_id));
            if (!origin) break;
            if (origin === loader) return true;
            const upstream = origin.inputs?.find((slot) => slot.type === "MODEL" && slot.link != null);
            link = upstream ? getLink(graph, upstream.link) : null;
          }
          return false;
        });
        if (byModel.length) return byModel;

        // Fallback: a loader group (visual group) usually holds its own samplers.
        const groups = graph.groups ?? graph._groups ?? [];
        const b = loader.pos && groups.find((group) => {
          const box = group.bounding ?? group._bounding;
          if (!box) return false;
          return (
            loader.pos[0] >= box[0] && loader.pos[1] >= box[1] &&
            loader.pos[0] <= box[0] + box[2] && loader.pos[1] <= box[1] + box[3]
          );
        });
        if (!b) return [];
        const box = b.bounding ?? b._bounding;
        return samplers.filter((sampler) => {
          const p = sampler.pos;
          return p && p[0] >= box[0] && p[1] >= box[1] && p[0] <= box[0] + box[2] && p[1] <= box[1] + box[3];
        });
      }

      function groupTitleForNode(graph, target) {
        const groups = graph.groups ?? graph._groups ?? [];
        const found = groups.find((group) => {
          const b = group.bounding ?? group._bounding;
          if (!b) return false;
          return (
            target.pos[0] >= b[0] && target.pos[1] >= b[1] &&
            target.pos[0] <= b[0] + b[2] && target.pos[1] <= b[1] + b[3]
          );
        });
        return found?.title ?? null;
      }

      async function importFromWorkflow() {
        const imported = {};
        const graphs = allGraphs();
        const upscaleLoaders = graphs.flatMap((graph) => (graph._nodes ?? []).filter((n) => n.type === "UpscaleModelLoader"));
        const singleUpscale = upscaleLoaders.length === 1
          ? String(findWidget(upscaleLoaders[0], ["model_name"])?.value ?? "")
          : "";
        for (const graph of graphs) {
          for (const loader of (graph._nodes ?? []).filter(isLoader)) {
            const model = loaderModelName(loader);
            if (!model) continue;
            const samplers = samplersForLoader(graph, loader);
            if (!samplers.length) continue;
            const name =
              groupTitleForNode(graph, loader) ||
              loader.title ||
              model.replace(/\.safetensors$/i, "");
            if (imported[name]) continue;

            const ordered = [...samplers].sort((a, b) => {
              const da = Number(findWidget(a, ["denoise"])?.value ?? 1);
              const db = Number(findWidget(b, ["denoise"])?.value ?? 1);
              return db - da;
            });
            const base = ordered[0];
            const hires = ordered[1];
            const read = (sampler, defaults) => {
              if (!sampler) return defaults;
              return {
                steps: Number(findWidget(sampler, ["steps"])?.value ?? defaults.steps),
                cfg: Number(findWidget(sampler, ["cfg"])?.value ?? defaults.cfg),
                sampler: String(findWidget(sampler, ["sampler_name"])?.value ?? defaults.sampler),
                scheduler: String(findWidget(sampler, ["scheduler"])?.value ?? defaults.scheduler),
                denoise: Number(findWidget(sampler, ["denoise"])?.value ?? defaults.denoise),
              };
            };
            imported[name] = {
              ...DEFAULT_PRESET,
              source: loader.type.includes("Checkpoint") ? "checkpoint" : "model",
              model,
              sage_attention: String(findWidget(loader, ["sage_attention"])?.value ?? "auto"),
              torch_compile: Boolean(findWidget(loader, ["torch_compile"])?.value ?? false),
              base: read(base, DEFAULT_PRESET.base),
              hires: { enabled: Boolean(hires), ...read(hires, DEFAULT_PRESET.hires) },
              upscale_model: singleUpscale,
            };
          }
        }

        const names = Object.keys(imported);
        if (!names.length) {
          setStatus("No model loaders with samplers found in this workflow.", "error");
          return;
        }
        for (const [name, preset] of Object.entries(imported)) {
          await savePreset(name, preset);
        }
        state.presets = await getPresets();
        if (presetWidget.options) presetWidget.options.values = Object.keys(state.presets);
        const firstName = names[0];
        presetWidget.value = firstName;
        state.creatingNew = false;
        await loadPreset(firstName);
        setStatus(`Imported ${names.length} preset(s). Check the values.`);
      }

      // ---- wiring -----------------------------------------------------------
      bindInputs();
      bindButtons();
      window.addEventListener("pointerdown", onPickPointerDown, true);
      refreshAutoSaveButton();
      refreshTargets();

      node.addDOMWidget("model_presets_panel", "div", container, { serialize: false });

      (async () => {
        try {
          [state.presets, state.options] = await Promise.all([getPresets(), getOptions()]);
          const names = Object.keys(state.presets);
          if (presetWidget.options) {
            presetWidget.options.values = names.length ? names : ["Default"];
          }
          updateSamplerSelects();
          if (!names.length) {
            fillPanel(DEFAULT_PRESET);
            nameInput.value = "Default";
            presetWidget.value = "Default";
            setStatus("No presets yet: fill the fields and press Save Preset.");
          } else {
            if (!names.includes(presetWidget.value)) {
              const missing = presetWidget.value;
              presetWidget.value = names[0];
              setStatus(`Preset "${missing}" not found; selected "${names[0]}".`, "warn");
            }
            await loadPreset(presetWidget.value);
          }
        } catch (error) {
          console.error("ModelPresets: init failed", error);
          setStatus(`Init failed: ${error.message}`, "error");
        }
      })();

      // Poll the preset dropdown: ComfyUI does not reliably fire callbacks for
      // programmatic widget changes.
      pollId = setInterval(() => {
        if (state.isLoading) return;
        if (presetWidget.value !== state.loadedPreset) {
          state.creatingNew = false;
          loadPreset(presetWidget.value);
        }
      }, POLL_INTERVAL);

      const originalConfigure = node.onConfigure;
      node.onConfigure = function () {
        const result = originalConfigure?.apply(this, arguments);
        setTimeout(() => {
          if (state.loadedPreset !== presetWidget.value) {
            loadPreset(presetWidget.value);
          } else {
            applyWithRetry("workflow load");
          }
        }, 100);
        return result;
      };

      const originalRemoved = node.onRemoved;
      node.onRemoved = function () {
        clearInterval(pollId);
        window.removeEventListener("pointerdown", onPickPointerDown, true);
        clearTimeout(autoSaveTimer);
        applyRetries.forEach(clearTimeout);
        return originalRemoved?.apply(this, arguments);
      };

      node._modelPresetsFlush = flushBeforeQueue;
      node._modelPresetsApply = () => applyCurrent("queue");
      return result;
    };
  },

  async beforeQueued() {
    const nodes = [];
    for (const graph of allGraphs()) {
      for (const node of graph._nodes ?? []) {
        if (node.type === NODE_TYPE) nodes.push(node);
      }
    }
    for (const node of nodes) {
      try {
        await node._modelPresetsFlush?.();
      } catch (error) {
        console.warn("ModelPresets: flush before queue failed", error);
      }
    }
  },
});
