import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const AUTO_PREFIX = "Auto download";
const DOWNLOAD_WIDGETS = {
  controlnet: [
    { nodeType: "SDMLX_ControlNetUnionLoader", widgetName: "control_net_name" },
  ],
  ipadapter: [
    { nodeType: "SDMLX_IPAdapterLoader", widgetName: "ipadapter_name" },
    { nodeType: "SDMLX_ApplyIPAdapterFaceIDAIO", widgetName: "ipadapter_name" },
  ],
  clip_vision: [
    { nodeType: "SDMLX_CLIPVisionLoader", widgetName: "clip_vision_name" },
    { nodeType: "SDMLX_ApplyIPAdapterFaceIDAIO", widgetName: "clip_vision_name" },
  ],
};
const NODE_TARGETS = Object.values(DOWNLOAD_WIDGETS).flat();

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function isAutoValue(value) {
  return value === "" || value === null || value === undefined
    || value === "auto"
    || (typeof value === "string" && value.startsWith(AUTO_PREFIX));
}

function concreteValues(values) {
  return (values || []).filter((value) => !isAutoValue(value));
}

function setWidgetValues(widget, values) {
  if (!widget || !values?.length) return;
  widget.options = widget.options || {};
  widget.options.values = values;
  widget.values = values;
}

async function fetchWidgetValues(nodeType, widgetName) {
  try {
    const response = await api.fetchApi(`/object_info/${nodeType}`);
    const data = await response.json();
    return data?.[nodeType]?.input?.required?.[widgetName]?.[0] || [];
  } catch (error) {
    console.warn(`SDMLX: Could not refresh ${nodeType}.${widgetName}`, error);
    return [];
  }
}

async function stabilizeNodeWidget(node, target, preferredValue = null) {
  const widget = widgetByName(node, target.widgetName);
  if (!widget) return;

  const values = await fetchWidgetValues(target.nodeType, target.widgetName);
  if (!values.length) return;

  const currentValue = widget.value;
  const resolvedValues = preferredValue && !values.includes(preferredValue)
    ? [...values, preferredValue]
    : values;
  const concrete = concreteValues(resolvedValues);
  const replacement = preferredValue || concrete[0] || resolvedValues[0];

  setWidgetValues(widget, resolvedValues);
  if (replacement && (isAutoValue(currentValue) || !resolvedValues.includes(currentValue))) {
    widget.value = replacement;
    widget.callback?.(widget.value, app.canvas, node);
  }
  node.setDirtyCanvas?.(true, true);
}

function stabilizeNode(node) {
  for (const target of NODE_TARGETS) {
    if (node.type === target.nodeType) {
      stabilizeNodeWidget(node, target);
    }
  }
}

async function refreshDownloadedModel(event) {
  const detail = event?.detail || {};
  const targets = DOWNLOAD_WIDGETS[detail.folder_name] || [];
  const localName = detail.local_name;
  if (!targets.length || !localName) return;

  for (const target of targets) {
    for (const node of app.graph?._nodes || []) {
      if (node.type !== target.nodeType) continue;
      await stabilizeNodeWidget(node, target, localName);
    }
  }
}

api.addEventListener("sdmlx_model_downloaded", refreshDownloadedModel);

app.registerExtension({
  name: "sdmlx.ModelDownloads",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_TARGETS.some((target) => target.nodeType === nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setTimeout(() => stabilizeNode(this), 0);
      return result;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setTimeout(() => stabilizeNode(this), 50);
      return result;
    };
  },
});
