import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set(["SDMLX_KSampler", "SDMLX_HiresFix"]);

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
  widget.tooltip = disabled ? tooltip : widget.sdmlxOriginalTooltip;
  if (widget.inputEl) {
    widget.inputEl.disabled = disabled;
    widget.inputEl.readOnly = disabled;
  }
}

function updateKSamplerWidgets(node) {
  const advancedConnected = inputConnected(inputByName(node, "spectrum_acceleration_advanced"));
  setWidgetDisabled(
    widgetByName(node, "spectrum_acceleration"),
    advancedConnected,
    "Controlled by Spectrum Advanced input"
  );
  node.setDirtyCanvas?.(true, true);
}

function stabilize(node) {
  const spectrumWidget = widgetByName(node, "spectrum_acceleration");
  if (spectrumWidget && !spectrumWidget.sdmlxSpectrumSamplerHooked) {
    const originalCallback = spectrumWidget.callback;
    spectrumWidget.callback = function (...args) {
      const result = originalCallback?.apply(this, args);
      setTimeout(() => updateKSamplerWidgets(node), 0);
      return result;
    };
    spectrumWidget.sdmlxSpectrumSamplerHooked = true;
  }
  updateKSamplerWidgets(node);
}

app.registerExtension({
  name: "sdmlx.SpectrumSwitch",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData.name)) return;

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
      setTimeout(() => updateKSamplerWidgets(this), 0);
      return result;
    };
  },
});
