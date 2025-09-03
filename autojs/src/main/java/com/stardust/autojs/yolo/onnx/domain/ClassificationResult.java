package com.stardust.autojs.yolo.onnx.domain;

public class ClassificationResult {
    private String label;
    private Integer clsId;
    private float confidence;

    public ClassificationResult() {
    }

    public ClassificationResult(String label, Integer clsId, float confidence) {
        this.label = label;
        this.clsId = clsId;
        this.confidence = confidence;
    }

    // getter 和 setter 方法保持不变
    public String getLabel() { return label; }
    public void setLabel(String label) { this.label = label; }
    
    public Integer getClsId() { return clsId; }
    public void setClsId(Integer clsId) { this.clsId = clsId; }
    
    public float getConfidence() { return confidence; }
    public void setConfidence(float confidence) { this.confidence = confidence; }

    @Override
    public String toString() {
        return "ClassificationResult{" +
                "label='" + label + '\'' +
                ", clsId=" + clsId +
                ", confidence=" + confidence +
                '}';
    }
}
