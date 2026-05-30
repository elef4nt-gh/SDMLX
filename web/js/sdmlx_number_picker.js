import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLX_NumberPicker";

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function parseValues(text) {
  const values = [];
  const seen = new Set();
  for (const raw of String(text || "").split(/[\s,;]+/)) {
    const label = raw.trim();
    if (!label || seen.has(label)) continue;
    const value = Number(label);
    if (!Number.isFinite(value)) continue;
    values.push(label);
    seen.add(label);
  }
  return values.length ? values : ["0"];
}

function setComboValues(widget, values) {
  if (!widget) return;
  widget.options = widget.options || {};
  widget.options.values = values;
  widget.values = values;
  if (!values.includes(String(widget.value))) {
    widget.value = values[0];
  }
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!widget.sdmlxOriginal) {
    widget.sdmlxOriginal = {
      type: widget.type,
      computeSize: widget.computeSize,
      disabled: widget.disabled,
      hidden: widget.hidden,
      optionsHidden: widget.options?.hidden,
      serialize: widget.serialize,
    };
  }
  widget.options = widget.options || {};
  if (visible) {
    widget.type = widget.sdmlxOriginal.type;
    widget.computeSize = widget.sdmlxOriginal.computeSize;
    widget.disabled = widget.sdmlxOriginal.disabled;
    widget.hidden = widget.sdmlxOriginal.hidden;
    widget.options.hidden = widget.sdmlxOriginal.optionsHidden;
    widget.serialize = widget.sdmlxOriginal.serialize;
  } else {
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.disabled = true;
    widget.hidden = true;
    widget.options.hidden = true;
    widget.serialize = true;
  }
}

function refreshSize(node) {
  const size = node.computeSize?.();
  const visibleWidgets = (node.widgets || []).filter((widget) => {
    return widget.type !== "hidden" && widget.hidden !== true && widget.options?.hidden !== true;
  }).length;
  const rowHeight = globalThis.LiteGraph?.NODE_WIDGET_HEIGHT || 20;
  const minHeight = 58 + visibleWidgets * (rowHeight + 4);
  if (size) {
    node.size = node.size || [size[0], size[1]];
    node.size[0] = Math.max(node.size[0], size[0], 210);
    node.size[1] = Math.max(88, size[1], minHeight);
  }
  node.setDirtyCanvas?.(true, true);
}

function updateNumberPicker(node) {
  const selectedWidget = widgetByName(node, "selected");
  const valuesWidget = widgetByName(node, "values");
  const editWidget = widgetByName(node, "edit_values");
  setComboValues(selectedWidget, parseValues(valuesWidget?.value));
  setWidgetVisible(valuesWidget, editWidget?.value === true);
  refreshSize(node);
}

function hookWidget(node, name) {
  const widget = widgetByName(node, name);
  if (!widget || widget.sdmlxNumberPickerHooked) return;
  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    setTimeout(() => updateNumberPicker(node), 0);
    return result;
  };
  widget.sdmlxNumberPickerHooked = true;
}

function stabilize(node) {
  hookWidget(node, "values");
  hookWidget(node, "edit_values");
  updateNumberPicker(node);
}

app.registerExtension({
  name: "sdmlx.NumberPicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      setTimeout(() => stabilize(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      this.serialize_widgets = true;
      setTimeout(() => stabilize(this), 50);
      return result;
    };
  },
});
