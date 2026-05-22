import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLX_LoraLoader";

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function inputByName(node, name) {
  return node.inputs?.find((input) => input.name === name);
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
  widget.tooltip = disabled
    ? tooltip
    : widget.sdmlxOriginalTooltip;
  if (widget.inputEl) {
    widget.inputEl.disabled = disabled;
    widget.inputEl.readOnly = disabled;
  }
}

function updateLoraLoaderWidgets(node) {
  const schedulerConnected = inputConnected(inputByName(node, "lora_scheduler"));
  const enabled = widgetByName(node, "enabled")?.value !== false;
  setWidgetDisabled(widgetByName(node, "lora_name"), !enabled, "LoRA loader disabled");
  setWidgetDisabled(
    widgetByName(node, "strength"),
    !enabled || schedulerConnected,
    enabled ? "Controlled by lora_scheduler" : "LoRA loader disabled"
  );
  node.setDirtyCanvas?.(true, true);
}

function hookWidgetCallbacks(node) {
  const widget = widgetByName(node, "enabled");
  if (!widget || widget.sdmlxHooked) return;
  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    setTimeout(() => updateLoraLoaderWidgets(node), 0);
    return result;
  };
  widget.sdmlxHooked = true;
}

function stabilize(node) {
  hookWidgetCallbacks(node);
  updateLoraLoaderWidgets(node);
}

app.registerExtension({
  name: "sdmlx.LoraLoader",
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

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = onConnectionsChange?.apply(this, arguments);
      setTimeout(() => updateLoraLoaderWidgets(this), 0);
      return result;
    };
  },
});
