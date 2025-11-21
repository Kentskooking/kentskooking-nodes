import { app } from "../../scripts/app.js";

const TARGET_NODE_NAMES = new Set([
    "VideoWaveController",
    "ImageWaveController",
    "InterpolationWaveController",
    "VideoIterativeSampler",
    "ImageIterativeSampler",
    "VideoInterpolateSampler",
    "WaveVisualizer",
    "WaveStyleModelApply",
    "WaveClipVisionEncode",
    "WaveIPAdapterController",
    "WaveIPAdapterAdvanced",
    "WaveControlNetApply",
    "WaveControlNetController",
    "IterativeCheckpointController",
    "CheckpointPreviewLoader",
]);

let desiredWidth = null;
const FALLBACK_WIDTH = 360;

function determineDesiredWidth(nodeType) {
    if (desiredWidth !== null) {
        return;
    }
    try {
        const instance = new nodeType();
        if (instance?.size?.[0]) {
            desiredWidth = instance.size[0];
        } else if (instance?.computeSize) {
            const computed = instance.computeSize();
            if (Array.isArray(computed) && computed[0]) {
                desiredWidth = computed[0];
            }
        }
        instance?.onRemoved?.();
    } catch (err) {
        console.warn("[kentskooking-width] Unable to sample advanced node width:", err);
    }
    if (!desiredWidth) {
        desiredWidth = FALLBACK_WIDTH;
    }
}

function enforceWidth(node) {
    if (!node?.size || desiredWidth === null) {
        return;
    }
    if (node.size[0] >= desiredWidth) {
        return;
    }
    node.size[0] = desiredWidth;
    if (typeof node.onResize === "function") {
        node.onResize(node.size);
    }
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "kentskooking.nodeWidth",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData?.name || !TARGET_NODE_NAMES.has(nodeData.name)) {
            if (nodeData?.name === "VideoWaveController") {
                determineDesiredWidth(nodeType);
            }
            return;
        }

        if (nodeData.name === "VideoWaveController") {
            determineDesiredWidth(nodeType);
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            if (typeof originalOnNodeCreated === "function") {
                originalOnNodeCreated.apply(this, args);
            }
            if (nodeData.name === "VideoWaveController" && desiredWidth === null) {
                desiredWidth = this?.size?.[0] ?? desiredWidth;
            }
            enforceWidth(this);
        };
    },
});