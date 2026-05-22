import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLX_ApplyControlNet";
const CONTROL_TYPES = [
  "pose",
  "depth",
  "soft edge to scribble",
  "line to canny",
  "normal",
  "segment",
  "tile",
  "repaint",
];

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function setComboValues(widget, values, defaultValue) {
  if (!widget) return;
  const aliases = {
    "soft edge / scribble": "soft edge to scribble",
    "line / canny": "line to canny",
  };
  if (aliases[widget.value]) {
    widget.value = aliases[widget.value];
  }
  widget.options = widget.options || {};
  widget.options.values = values;
  widget.values = values;
  if (!values.includes(widget.value)) {
    widget.value = defaultValue;
  }
}

function stabilize(node) {
  setComboValues(widgetByName(node, "control_type"), CONTROL_TYPES, "line to canny");
  node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
  name: "sdmlx.ControlNet",
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
