import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLX_LoraSchedule";
const CURVES_BY_MODE = {
  "blend in": ["linear", "progressive", "progressive fast", "s-curve"],
  "blend out": ["linear", "degressive", "degressive fast", "s-curve"],
  bell: ["positive", "negative"],
};

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
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
    widget.serialize = false;
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
    node.size[0] = Math.max(node.size[0], size[0], 300);
    node.size[1] = Math.max(130, size[1], minHeight);
  }
  node.setDirtyCanvas?.(true, true);
}

function setComboValues(widget, values) {
  if (!widget) return;
  widget.options = widget.options || {};
  widget.options.values = values;
  widget.values = values;
  if (!values.includes(widget.value)) {
    widget.value = values[0];
  }
}

function updateScheduleWidgets(node) {
  const modeWidget = widgetByName(node, "mode");
  const curveWidget = widgetByName(node, "curve");
  const advancedWidget = widgetByName(node, "advanced");
  const mode = modeWidget?.value || "blend in";
  const curves = CURVES_BY_MODE[mode] || CURVES_BY_MODE["blend in"];
  setComboValues(curveWidget, curves);

  const advanced = Boolean(advancedWidget?.value);
  setWidgetVisible(widgetByName(node, "start_percent"), advanced);
  setWidgetVisible(widgetByName(node, "end_percent"), advanced);
  refreshSize(node);
}

function hookWidget(node, name) {
  const widget = widgetByName(node, name);
  if (!widget || widget.sdmlxScheduleHooked) return;
  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    setTimeout(() => updateScheduleWidgets(node), 0);
    return result;
  };
  widget.sdmlxScheduleHooked = true;
}

function stabilize(node) {
  hookWidget(node, "mode");
  hookWidget(node, "advanced");
  updateScheduleWidgets(node);
}

app.registerExtension({
  name: "sdmlx.LoraSchedule",
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
    nodeType.prototype.onConfigure = function (config) {
      const result = onConfigure?.apply(this, arguments);
      this.serialize_widgets = true;
      setTimeout(() => stabilize(this), 50);
      return result;
    };
  },
});
