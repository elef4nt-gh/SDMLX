import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLX_LoraSchedule";
const CURVES_BY_MODE = {
  "blend in": ["linear", "progressive", "progressive fast", "s-curve"],
  "blend out": ["linear", "degressive", "degressive fast", "s-curve"],
  bell: ["positive", "negative"],
};
const MIN_NODE_WIDTH = 300;

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

function compactNodeSize(node, forceMinimumWidth = false) {
  const size = compactSize(node);
  node.size = node.size || [size[0], size[1]];
  node.size[0] = forceMinimumWidth ? MIN_NODE_WIDTH : Math.max(node.size[0] ?? MIN_NODE_WIDTH, MIN_NODE_WIDTH);
  node.size[1] = size[1];
  node.setDirtyCanvas?.(true, true);
}

function visibleWidgetCount(node) {
  return (node.widgets || []).filter((widget) => {
    return widget.type !== "hidden" && widget.hidden !== true && widget.options?.hidden !== true;
  }).length;
}

function compactSize(node) {
  const rowHeight = globalThis.LiteGraph?.NODE_WIDGET_HEIGHT || 20;
  const height = Math.max(96, 46 + visibleWidgetCount(node) * (rowHeight + 4));
  return [MIN_NODE_WIDTH, height];
}

function configuredSize(config) {
  const size = config?.size;
  if (!size) return null;
  const width = Number(size[0]);
  const height = Number(size[1]);
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
  return [Math.max(MIN_NODE_WIDTH, width), Math.max(1, height)];
}

function restoreConfiguredSize(node, size) {
  const contentSize = compactSize(node);
  const width = Math.max(MIN_NODE_WIDTH, size?.[0] ?? node.size?.[0] ?? contentSize[0]);
  const configuredHeight = Number(size?.[1]);
  const maxReasonableHeight = Math.max(contentSize[1] + 40, contentSize[1] * 1.35);
  const height = Number.isFinite(configuredHeight) && configuredHeight <= maxReasonableHeight
    ? Math.max(contentSize[1], configuredHeight)
    : contentSize[1];
  node.size = [width, height];
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

function updateScheduleWidgets(node, resize = false, forceMinimumWidth = false) {
  const modeWidget = widgetByName(node, "mode");
  const curveWidget = widgetByName(node, "curve");
  const advancedWidget = widgetByName(node, "advanced");
  const mode = modeWidget?.value || "blend in";
  const curves = CURVES_BY_MODE[mode] || CURVES_BY_MODE["blend in"];
  setComboValues(curveWidget, curves);

  const advanced = Boolean(advancedWidget?.value);
  setWidgetVisible(widgetByName(node, "start_percent"), advanced);
  setWidgetVisible(widgetByName(node, "end_percent"), advanced);
  if (resize) {
    compactNodeSize(node, forceMinimumWidth);
  }
}

function hookWidget(node, name) {
  const widget = widgetByName(node, name);
  if (!widget || widget.sdmlxScheduleHooked) return;
  const originalCallback = widget.callback;
  widget.callback = function (...args) {
    const result = originalCallback?.apply(this, args);
    setTimeout(() => updateScheduleWidgets(node, true), 0);
    return result;
  };
  widget.sdmlxScheduleHooked = true;
}

function stabilize(node, resize = false, forceMinimumWidth = false) {
  hookWidget(node, "mode");
  hookWidget(node, "advanced");
  updateScheduleWidgets(node, resize, forceMinimumWidth);
}

app.registerExtension({
  name: "sdmlx.LoraSchedule",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    nodeType.prototype.computeSize = function () {
      return compactSize(this);
    };

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      this.serialize_widgets = true;
      this.sdmlxConfiguredFromWorkflow = false;
      setTimeout(() => {
        if (!this.sdmlxConfiguredFromWorkflow) stabilize(this, true, true);
      }, 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (config) {
      const size = configuredSize(config);
      const result = onConfigure?.apply(this, arguments);
      this.serialize_widgets = true;
      this.sdmlxConfiguredFromWorkflow = true;
      stabilize(this, false);
      restoreConfiguredSize(this, size);
      setTimeout(() => {
        stabilize(this, false);
        restoreConfiguredSize(this, size);
      }, 0);
      setTimeout(() => {
        stabilize(this, false);
        restoreConfiguredSize(this, size);
      }, 50);
      return result;
    };
  },
});
