import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLXFlux2KleinEnhancedEditSampler";
const MAX_REFERENCE_SLOTS = 8;

function referenceImageName(index) {
  return `reference_image_${index}`;
}

function subjectMaskName(index) {
  return `subject_mask_${index}`;
}

function imageLabel(index) {
  return `image ${index}`;
}

function maskLabel(index) {
  return `mask ${index}`;
}

function dynamicKind(name) {
  if (/^reference_image_[1-8]$/.test(String(name))) return "image";
  if (/^subject_mask_[1-8]$/.test(String(name))) return "mask";
  return null;
}

function dynamicIndex(name) {
  const match = String(name ?? "").match(/_(\d+)$/);
  return match ? Number.parseInt(match[1], 10) : 0;
}

function inputByName(node, name) {
  return node.inputs?.find((input) => input.name === name);
}

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function inputIndexByName(node, name) {
  return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function inputConnected(input) {
  return Boolean(input && (input.link != null || input.links?.length));
}

function setWidgetDisabled(widget, disabled, tooltip = "Disabled") {
  if (!widget) return;
  if (widget.sdmlxOriginalDisabled === undefined) {
    widget.sdmlxOriginalDisabled = Boolean(widget.disabled);
    widget.sdmlxOriginalOptionsDisabled = Boolean(widget.options?.disabled);
    widget.sdmlxOriginalTooltip = widget.tooltip;
  }
  widget.disabled = disabled || widget.sdmlxOriginalDisabled;
  widget.options = widget.options || {};
  widget.options.disabled = disabled || widget.sdmlxOriginalOptionsDisabled;
  widget.tooltip = disabled ? tooltip : widget.sdmlxOriginalTooltip;
  if (widget.inputEl) {
    widget.inputEl.disabled = disabled;
    widget.inputEl.readOnly = disabled;
  }
}

function updatePresetWidget(node) {
  const advancedConnected = inputConnected(inputByName(node, "enhancer_advanced"));
  setWidgetDisabled(
    widgetByName(node, "enhance_preset"),
    advancedConnected,
    "Controlled by FLUX.2 Klein Enhancer Advanced input"
  );
}

function labelInput(input, label) {
  if (!input) return;
  input.label = label;
  input.localized_name = label;
}

function labelDynamicInputs(node) {
  for (const input of node.inputs ?? []) {
    const index = dynamicIndex(input.name);
    if (!index) continue;
    if (dynamicKind(input.name) === "image") labelInput(input, imageLabel(index));
    if (dynamicKind(input.name) === "mask") labelInput(input, maskLabel(index));
  }
}

function pairConnected(node, index) {
  return (
    inputConnected(inputByName(node, referenceImageName(index))) ||
    inputConnected(inputByName(node, subjectMaskName(index)))
  );
}

function highestConnectedPair(node) {
  let highest = 0;
  for (let index = 1; index <= MAX_REFERENCE_SLOTS; index++) {
    if (pairConnected(node, index)) highest = index;
  }
  return highest;
}

function desiredPairCount(node) {
  const highest = highestConnectedPair(node);
  return Math.min(MAX_REFERENCE_SLOTS, Math.max(1, highest + 1));
}

function ensureInput(node, name, type, label) {
  let input = inputByName(node, name);
  if (!input) {
    node.addInput(name, type, { label });
    input = inputByName(node, name);
  }
  if (input) {
    input.type = type;
    labelInput(input, label);
  }
  return input;
}

function removeInputIfSafe(node, name) {
  const index = inputIndexByName(node, name);
  if (index < 0) return;
  const input = node.inputs[index];
  if (inputConnected(input)) return;
  node.removeInput(index);
}

function linkForInput(node, input) {
  if (input?.link == null || !node.graph?.links) return null;
  return node.graph.links[input.link] ?? null;
}

function updateInputLinkSlots(node) {
  for (let index = 0; index < (node.inputs?.length ?? 0); index++) {
    const link = linkForInput(node, node.inputs[index]);
    if (link) link.target_slot = index;
  }
}

function reorderReferenceInputs(node) {
  if (!node.inputs?.length) return;
  const base = [];
  const images = [];
  const masks = [];

  for (const input of node.inputs) {
    const kind = dynamicKind(input.name);
    if (kind === "image") images.push(input);
    else if (kind === "mask") masks.push(input);
    else base.push(input);
  }

  images.sort((a, b) => dynamicIndex(a.name) - dynamicIndex(b.name));
  masks.sort((a, b) => dynamicIndex(a.name) - dynamicIndex(b.name));
  node.inputs = [...base, ...images, ...masks];
  updateInputLinkSlots(node);
}

function resizeForInputs(node) {
  if (typeof node.computeSize !== "function") return;
  const computed = node.computeSize();
  if (!computed || !node.size) return;
  const width = Math.max(node.size[0] ?? 0, computed[0] ?? 0);
  const height = computed[1] ?? node.size[1] ?? 0;
  if (typeof node.setSize === "function") {
    node.setSize([width, height]);
  } else {
    node.size = [width, height];
  }
}

function stabilizeReferenceInputs(node) {
  if (!node.inputs || node.sdmlxFlux2EnhancedStabilizing) return;
  node.sdmlxFlux2EnhancedStabilizing = true;
  try {
    const desired = desiredPairCount(node);

    for (let index = 1; index <= desired; index++) {
      ensureInput(node, referenceImageName(index), "IMAGE", imageLabel(index));
      ensureInput(node, subjectMaskName(index), "MASK", maskLabel(index));
    }

    for (let index = MAX_REFERENCE_SLOTS; index > desired; index--) {
      removeInputIfSafe(node, referenceImageName(index));
      removeInputIfSafe(node, subjectMaskName(index));
    }

    labelDynamicInputs(node);
    reorderReferenceInputs(node);
    updatePresetWidget(node);
    resizeForInputs(node);
    node.setDirtyCanvas?.(true, true);
  } finally {
    node.sdmlxFlux2EnhancedStabilizing = false;
  }
}

app.registerExtension({
  name: "sdmlx.Flux2EnhancedEditInputs",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setTimeout(() => stabilizeReferenceInputs(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setTimeout(() => stabilizeReferenceInputs(this), 0);
      setTimeout(() => stabilizeReferenceInputs(this), 50);
      return result;
    };

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = onConnectionsChange?.apply(this, arguments);
      setTimeout(() => stabilizeReferenceInputs(this), 0);
      return result;
    };
  },
});
