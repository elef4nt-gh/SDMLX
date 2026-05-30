import { app } from "../../../scripts/app.js";

const NODE_NAME = "SDMLX_MultiLoraLoader";
const LORA_SLOT_COUNT = 12;
const NONE_VALUES = new Set(["none", "None", "", null, undefined]);
const ROW_PREFIX = "sdmlx_lora_row_";
const STRENGTH_MIN = -4;
const STRENGTH_MAX = 4;
const STRENGTH_STEP = 0.05;
const MIN_NODE_WIDTH = 300;

function customRowHeight() {
  return (globalThis.LiteGraph?.NODE_WIDGET_HEIGHT || 20) + 4;
}

function widgetRadius(height) {
  return height * 0.5;
}

function widgetByName(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function isInternalWidgetInput(name) {
  return name === "slot_count" || /^(enabled|lora|strength)_\d+$/.test(String(name ?? ""));
}

function removeWidgetInputSockets(node) {
  if (!node.inputs?.length) return;
  for (let index = node.inputs.length - 1; index >= 0; index--) {
    const input = node.inputs[index];
    const name = input.widget?.name ?? input.name;
    if (!isInternalWidgetInput(name)) continue;
    if (typeof node.removeInput === "function") {
      node.removeInput(index);
    } else {
      node.inputs.splice(index, 1);
    }
  }
}

function isLoraSelected(node, index) {
  const widget = widgetByName(node, `lora_${index}`);
  return !NONE_VALUES.has(widget?.value);
}

function slotCount(node) {
  const widget = widgetByName(node, "slot_count");
  const value = Number.parseInt(widget?.value ?? 1, 10);
  return Math.min(LORA_SLOT_COUNT, Math.max(1, Number.isFinite(value) ? value : 1));
}

function setSlotCount(node, value) {
  const widget = widgetByName(node, "slot_count");
  if (!widget) return;
  widget.value = Math.min(LORA_SLOT_COUNT, Math.max(1, value));
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!widget.sdmlxOriginal) {
    widget.sdmlxOriginal = {
      type: widget.type,
      computeSize: widget.computeSize,
      hidden: widget.hidden,
      optionsHidden: widget.options?.hidden,
    };
  }
  widget.options = widget.options || {};
  if (visible) {
    widget.type = widget.sdmlxOriginal.type;
    widget.computeSize = widget.sdmlxOriginal.computeSize;
    widget.hidden = widget.sdmlxOriginal.hidden;
    widget.options.hidden = widget.sdmlxOriginal.optionsHidden;
  } else {
    widget.type = `sdmlx-hidden:${widget.sdmlxOriginal.type}`;
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
    widget.options.hidden = true;
  }
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function roundStrength(value) {
  return Math.round(value * 100) / 100;
}

function formatStrength(value) {
  return roundStrength(value).toFixed(2).replace(/\.?0+$/, "");
}

function widgetValues(widget) {
  const values = widget?.options?.values;
  if (Array.isArray(values)) return values;
  if (typeof values === "function") return values(widget) ?? [];
  return [];
}

function fitText(ctx, text, maxWidth) {
  const value = String(text ?? "");
  if (ctx.measureText(value).width <= maxWidth) return value;
  const suffix = "...";
  let left = 0;
  let right = value.length;
  while (left < right) {
    const mid = Math.ceil((left + right) / 2);
    if (ctx.measureText(value.slice(0, mid) + suffix).width <= maxWidth) {
      left = mid;
    } else {
      right = mid - 1;
    }
  }
  return value.slice(0, left) + suffix;
}

function roundedRect(ctx, x, y, width, height, radius) {
  if (ctx.roundRect) {
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, radius);
    return;
  }
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
}

function drawSlotToggle(ctx, x, y, width, height, enabled) {
  const radius = height / 2;
  roundedRect(ctx, x, y, width, height, radius);
  ctx.fillStyle = enabled ? "#22c55e" : LiteGraph.WIDGET_BGCOLOR;
  ctx.fill();
  ctx.strokeStyle = enabled ? "#22c55e" : LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.stroke();

  const knobSize = Math.max(8, height - 4);
  const knobX = enabled ? x + width - knobSize - 2 : x + 2;
  const knobY = y + (height - knobSize) / 2;
  ctx.beginPath();
  ctx.arc(knobX + knobSize / 2, knobY + knobSize / 2, knobSize / 2, 0, Math.PI * 2);
  ctx.fillStyle = "#ffffff";
  ctx.fill();
}

function drawStepperArrow(ctx, tipX, y, height, direction, disabled = false) {
  const arrowWidth = 10;
  const top = y + 5;
  const middle = y + height * 0.5;
  const bottom = y + height - 5;
  ctx.beginPath();
  if (direction < 0) {
    ctx.moveTo(tipX + arrowWidth, top);
    ctx.lineTo(tipX, middle);
    ctx.lineTo(tipX + arrowWidth, bottom);
  } else {
    ctx.moveTo(tipX - arrowWidth, top);
    ctx.lineTo(tipX, middle);
    ctx.lineTo(tipX - arrowWidth, bottom);
  }
  ctx.fillStyle = disabled ? LiteGraph.WIDGET_SECONDARY_TEXT_COLOR : LiteGraph.WIDGET_TEXT_COLOR || "#ffffff";
  ctx.fill();
}

function setLoraValue(node, index, value, event) {
  const widget = widgetByName(node, `lora_${index}`);
  if (!widget) return;
  widget.value = value;
  widget.callback?.(value, app.canvas, node, undefined, event);
  updateMultiLoraNode(node, true);
}

function setEnabledValue(node, index, value, event) {
  const widget = widgetByName(node, `enabled_${index}`);
  if (!widget) return;
  widget.value = !!value;
  widget.callback?.(widget.value, app.canvas, node, undefined, event);
  node.setDirtyCanvas?.(true, true);
}

function setStrengthValue(node, index, value, event) {
  const widget = widgetByName(node, `strength_${index}`);
  if (!widget) return;
  const next = roundStrength(clamp(Number(value), STRENGTH_MIN, STRENGTH_MAX));
  if (!Number.isFinite(next)) return;
  widget.value = next;
  widget.callback?.(next, app.canvas, node, undefined, event);
  node.setDirtyCanvas?.(true, true);
}

function showLoraChooser(event, node, index) {
  const widget = widgetByName(node, `lora_${index}`);
  const values = widgetValues(widget);
  if (!widget || !values.length || !globalThis.LiteGraph?.ContextMenu) return;
  const scale = Math.max(1, app.canvas?.ds?.scale ?? 1);
  new LiteGraph.ContextMenu(values, {
    event,
    title: "Choose LoRA",
    className: "dark",
    scale,
    callback: (value) => {
      if (value == null) return;
      const next = typeof value === "object" ? (value.value ?? value.content) : value;
      if (next == null) return;
      setLoraValue(node, index, String(next), event);
    },
  });
}

function nextAvailableSlot(node) {
  const visibleSlots = slotCount(node);
  for (let index = 1; index <= visibleSlots; index++) {
    if (!isLoraSelected(node, index)) return index;
  }
  return visibleSlots < LORA_SLOT_COUNT ? visibleSlots + 1 : null;
}

function addLoraAndChoose(event, node) {
  if (!globalLorasEnabled(node)) return;
  const targetSlot = nextAvailableSlot(node);
  if (!targetSlot) return;
  if (targetSlot > slotCount(node)) {
    setSlotCount(node, targetSlot);
  }
  updateMultiLoraNode(node, true);
  setTimeout(() => showLoraChooser(event, node, targetSlot), 0);
}

function globalLorasEnabled(node) {
  return widgetByName(node, "enabled")?.value !== false;
}

function drawSlotRow(ctx, node, index, width, posY, height, rowWidget) {
  rowWidget.sdmlxLastY = posY;
  const loraWidget = widgetByName(node, `lora_${index}`);
  const enabledWidget = widgetByName(node, `enabled_${index}`);
  const strengthWidget = widgetByName(node, `strength_${index}`);
  const selected = !NONE_VALUES.has(loraWidget?.value);
  const globalEnabled = globalLorasEnabled(node);
  const enabled = globalEnabled && enabledWidget?.value !== false;
  const strength = Number.parseFloat(strengthWidget?.value ?? 1);
  const margin = 15;
  const inner = 6;
  const rowX = margin;
  const rowY = posY + 2;
  const rowH = Math.max(20, height - 4);
  const rowW = Math.max(0, width - margin * 2);
  const toggleW = 26;
  const toggleH = 14;
  const toggleX = rowX + 7;
  const toggleY = rowY + (rowH - toggleH) / 2;
  const strengthW = selected ? 72 : 0;
  const strengthX = rowX + rowW - strengthW - inner;
  const loraX = toggleX + toggleW + inner + 2;
  const loraW = Math.max(40, (selected ? strengthX - inner : rowX + rowW - inner) - loraX);
  const alpha = app.canvas?.editor_alpha ?? 1;
  const rowAlpha = globalEnabled ? alpha : alpha * 0.45;

  ctx.save();
  ctx.globalAlpha = rowAlpha;
  roundedRect(ctx, rowX, rowY, rowW, rowH, widgetRadius(rowH));
  ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
  ctx.fill();
  ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
  ctx.stroke();

  drawSlotToggle(ctx, toggleX, toggleY, toggleW, toggleH, selected && enabled);

  if (selected && !enabled) ctx.globalAlpha = alpha * 0.45;
  ctx.fillStyle = selected ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const label = selected ? loraWidget.value : "choose lora";
  ctx.fillText(fitText(ctx, label, loraW), loraX, rowY + rowH / 2);

  rowWidget.sdmlxBounds = {
    toggle: [toggleX, rowY, toggleW, rowH],
    lora: [loraX, rowY, loraW, rowH],
    strengthDec: null,
    strengthValue: null,
    strengthInc: null,
  };

  if (selected) {
    ctx.globalAlpha = enabled ? alpha : alpha * 0.45;
    const arrowButtonW = 20;
    const valueX = strengthX + arrowButtonW;
    const valueW = strengthW - arrowButtonW * 2;
    const leftTipX = strengthX + 6;
    const rightTipX = strengthX + strengthW - 6;
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    drawStepperArrow(ctx, leftTipX, rowY, rowH, -1, !enabled);
    ctx.fillText(formatStrength(strength), valueX + valueW / 2, rowY + rowH / 2);
    drawStepperArrow(ctx, rightTipX, rowY, rowH, 1, !enabled);
    rowWidget.sdmlxBounds.strengthDec = [strengthX, rowY, arrowButtonW, rowH];
    rowWidget.sdmlxBounds.strengthValue = [valueX, rowY, valueW, rowH];
    rowWidget.sdmlxBounds.strengthInc = [valueX + valueW, rowY, arrowButtonW, rowH];
  }
  ctx.restore();
}

function pointInBounds(pos, bounds) {
  if (!bounds) return false;
  const [x, y, width, height] = bounds;
  return pos[0] >= x && pos[0] <= x + width && pos[1] >= y && pos[1] <= y + height;
}

function handleSlotRowMouse(event, pos, node, index, rowWidget) {
  if (event.type !== "pointerdown") return false;
  const bounds = rowWidget.sdmlxBounds ?? {};
  if (!globalLorasEnabled(node)) {
    return pointInBounds(pos, bounds.toggle)
      || pointInBounds(pos, bounds.lora)
      || pointInBounds(pos, bounds.strengthDec)
      || pointInBounds(pos, bounds.strengthValue)
      || pointInBounds(pos, bounds.strengthInc);
  }
  if (pointInBounds(pos, bounds.toggle)) {
    if (isLoraSelected(node, index)) {
      const widget = widgetByName(node, `enabled_${index}`);
      setEnabledValue(node, index, widget?.value === false, event);
    }
    return true;
  }
  if (pointInBounds(pos, bounds.strengthDec)) {
    const widget = widgetByName(node, `strength_${index}`);
    setStrengthValue(node, index, Number.parseFloat(widget?.value ?? 1) - STRENGTH_STEP, event);
    return true;
  }
  if (pointInBounds(pos, bounds.strengthInc)) {
    const widget = widgetByName(node, `strength_${index}`);
    setStrengthValue(node, index, Number.parseFloat(widget?.value ?? 1) + STRENGTH_STEP, event);
    return true;
  }
  if (pointInBounds(pos, bounds.strengthValue)) {
    const widget = widgetByName(node, `strength_${index}`);
    const current = widget?.value ?? 1;
    if (app.canvas?.prompt) {
      app.canvas.prompt("LoRA strength", current, (value) => {
        setStrengthValue(node, index, Number.parseFloat(value), event);
      }, event);
    } else {
      const value = globalThis.prompt?.("LoRA strength", current);
      if (value != null) setStrengthValue(node, index, Number.parseFloat(value), event);
    }
    return true;
  }
  if (pointInBounds(pos, bounds.lora)) {
    showLoraChooser(event, node, index);
    return true;
  }
  return false;
}

function makeSlotRowWidget(index) {
  return {
    name: `${ROW_PREFIX}${index}`,
    type: "custom",
    value: "",
    options: { serialize: false },
    serialize: false,
    computeSize(width) {
      return [width, customRowHeight()];
    },
    draw(ctx, node, width, posY, height) {
      drawSlotRow(ctx, node, index, width, posY, height, this);
    },
    mouse(event, pos, node) {
      return handleSlotRowMouse(event, pos, node, index, this);
    },
    serializeValue() {
      return undefined;
    },
  };
}

function ensureSlotRows(node) {
  if (!node.addCustomWidget) return false;
  node.sdmlxSlotRowWidgets ??= [];
  for (let index = 1; index <= LORA_SLOT_COUNT; index++) {
    if (node.sdmlxSlotRowWidgets[index]) continue;
    const rowWidget = makeSlotRowWidget(index);
    node.addCustomWidget(rowWidget);
    node.sdmlxSlotRowWidgets[index] = rowWidget;
  }
  return true;
}

function orderCustomWidgets(node) {
  const rows = (node.sdmlxSlotRowWidgets ?? []).filter(Boolean);
  const tail = [...rows, node.sdmlxAddLoraWidget].filter(Boolean);
  if (!tail.length) return;
  node.widgets = node.widgets.filter((widget) => !tail.includes(widget));
  node.widgets.push(...tail);
}

function slotWidgetNames(index) {
  return [
    `enabled_${index}`,
    `lora_${index}`,
    `strength_${index}`,
  ];
}

function rackRowCount(node) {
  const visibleSlots = slotCount(node);
  let rows = 0;
  for (let index = 1; index <= visibleSlots; index++) {
    if (isLoraSelected(node, index)) rows += 1;
  }
  if (visibleSlots < LORA_SLOT_COUNT) rows += 1;
  return Math.max(1, rows);
}

function compactHeight(node) {
  const rowHeight = customRowHeight();
  return Math.max(70, 46 + rackRowCount(node) * (rowHeight + 4));
}

function minimumSize(node) {
  return [MIN_NODE_WIDTH, compactHeight(node)];
}

function maximumUsefulHeight() {
  const rowHeight = customRowHeight();
  return Math.max(70, 46 + LORA_SLOT_COUNT * (rowHeight + 4));
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
  const contentSize = minimumSize(node);
  const width = Math.max(MIN_NODE_WIDTH, size?.[0] ?? node.size?.[0] ?? contentSize[0]);
  const configuredHeight = Number(size?.[1]);
  const maxReasonableHeight = maximumUsefulHeight() + 160;
  const height = Number.isFinite(configuredHeight) && configuredHeight <= maxReasonableHeight
    ? Math.max(contentSize[1], configuredHeight)
    : contentSize[1];
  node.size = [width, height];
  node.setDirtyCanvas?.(true, true);
}

function compactNodeSize(node, forceMinimumWidth = false) {
  const width = forceMinimumWidth ? MIN_NODE_WIDTH : Math.max(MIN_NODE_WIDTH, node.size?.[0] ?? MIN_NODE_WIDTH);
  const height = compactHeight(node);
  node.size = node.size || [width, compactHeight(node)];
  node.size[0] = width;
  node.size[1] = forceMinimumWidth ? height : Math.max(node.size[1] ?? height, height);
  node.setDirtyCanvas?.(true, true);
}

function ensureAddButton(node) {
  if (node.sdmlxAddLoraWidget) return;
  const onClick = (event) => addLoraAndChoose(event, node);
  if (node.addCustomWidget) {
    const button = {
      name: "sdmlx_add_lora",
      type: "custom",
      value: "",
      options: { serialize: false },
      serialize: false,
      computeSize(width) {
        return [width, customRowHeight()];
      },
      draw(ctx, node, width, posY, height) {
        const globalEnabled = globalLorasEnabled(node);
        const margin = 15;
        const rowX = margin;
        const rowY = posY + 2;
        const rowW = width - margin * 2;
        const rowH = Math.max(20, height - 4);
        const midY = rowY + rowH / 2;
        const plusX = rowX + 12;
        ctx.save();
        ctx.globalAlpha = globalEnabled ? (app.canvas?.editor_alpha ?? 1) : (app.canvas?.editor_alpha ?? 1) * 0.45;
        roundedRect(ctx, rowX, rowY, rowW, rowH, widgetRadius(rowH));
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.fill();
        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
        ctx.stroke();
        ctx.strokeStyle = "#62bd5b";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(plusX - 4, midY);
        ctx.lineTo(plusX + 4, midY);
        ctx.moveTo(plusX, midY - 4);
        ctx.lineTo(plusX, midY + 4);
        ctx.stroke();
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText("add lora", plusX + 12, midY);
        ctx.restore();
      },
      mouse(event) {
        if (event.type !== "pointerdown") return false;
        if (!globalLorasEnabled(node)) return true;
        onClick(event);
        return true;
      },
      serializeValue() {
        return undefined;
      },
    };
    node.addCustomWidget(button);
    node.sdmlxAddLoraWidget = button;
    return;
  }
  if (!node.addWidget) return;
  const button = node.addWidget("button", "✚ add lora", null, (_value, _canvas, _node, _pos, event) => onClick(event));
  button.serialize = false;
  node.sdmlxAddLoraWidget = button;
}

function updateLoraWidgets(node) {
  const visibleSlots = slotCount(node);
  setWidgetVisible(widgetByName(node, "slot_count"), false);
  const hasCustomRows = ensureSlotRows(node);

  for (let index = 1; index <= LORA_SLOT_COUNT; index++) {
    if (hasCustomRows) {
      for (const name of slotWidgetNames(index)) {
        setWidgetVisible(widgetByName(node, name), false);
      }
      setWidgetVisible(node.sdmlxSlotRowWidgets[index], index <= visibleSlots && isLoraSelected(node, index));
    } else {
      const showSelector = index <= visibleSlots;
      const showControls = showSelector && isLoraSelected(node, index);
      setWidgetVisible(widgetByName(node, `lora_${index}`), showSelector);
      for (const name of slotWidgetNames(index)) {
        if (name === `lora_${index}`) continue;
        setWidgetVisible(widgetByName(node, name), showControls);
      }
    }
  }

  ensureAddButton(node);
  orderCustomWidgets(node);
  setWidgetVisible(node.sdmlxAddLoraWidget, visibleSlots < LORA_SLOT_COUNT);
}

function hookLoraWidgetCallbacks(node) {
  for (let index = 1; index <= LORA_SLOT_COUNT; index++) {
    const widget = widgetByName(node, `lora_${index}`);
    if (!widget || widget.sdmlxHooked) continue;
    const originalCallback = widget.callback;
    widget.callback = function (...args) {
      const result = originalCallback?.apply(this, args);
      setTimeout(() => updateMultiLoraNode(node, true), 0);
      return result;
    };
    widget.sdmlxHooked = true;
  }
  const globalEnabled = widgetByName(node, "enabled");
  if (globalEnabled && !globalEnabled.sdmlxHooked) {
    const originalCallback = globalEnabled.callback;
    globalEnabled.callback = function (...args) {
      const result = originalCallback?.apply(this, args);
      setTimeout(() => updateMultiLoraNode(node, true), 0);
      return result;
    };
    globalEnabled.sdmlxHooked = true;
  }
}

function updateMultiLoraNode(node, resize = false, forceMinimumWidth = false) {
  removeWidgetInputSockets(node);
  updateLoraWidgets(node);
  if (resize) {
    compactNodeSize(node, forceMinimumWidth);
  }
}

function stabilize(node, resize = false, forceMinimumWidth = false) {
  hookLoraWidgetCallbacks(node);
  updateMultiLoraNode(node, resize, forceMinimumWidth);
}

app.registerExtension({
  name: "sdmlx.MultiLoraLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    nodeType.prototype.computeSize = function () {
      return minimumSize(this);
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

    const onConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = onConnectionsChange?.apply(this, arguments);
      setTimeout(() => updateMultiLoraNode(this, false), 0);
      return result;
    };
  },
});
