package com.stardust.autojs.yolo;

import android.util.Log;
import com.stardust.autojs.yolo.onnx.domain.ClassificationResult;
import com.stardust.autojs.yolo.onnx.domain.DetectResult;
import org.opencv.core.Mat;
import java.util.Collections;
import java.util.List;

public class BaseYoloInstance extends YoloInstance {

    private final YoloPredictor predictor;

    public BaseYoloInstance(YoloPredictor predictor) {
        this.predictor = predictor;
    }

    @Override
    public YoloPredictor getPredictor() {
        return predictor;
    }

    @Override
    public List<DetectResult> predictYolo(Mat image) {
        try {
            return predictor.predictYolo(image);
        } catch (Exception e) {
            Log.e("BaseYoloInstance", "predictYolo failed", e);
            return Collections.emptyList();
        }
    }

    @Override
    public List<ClassificationResult> predictClassification(Mat image) {
        try {
            return predictor.predictClassification(image);
        } catch (Exception e) {
            Log.e("BaseYoloInstance", "predictClassification failed", e);
            return Collections.emptyList();
        }
    }

    @Override
    public void setConfThreshold(float confThreshold) {
        predictor.setConfThreshold(confThreshold);
    }

    @Override
    public void setClassificationThreshold(float classificationThreshold) {
        predictor.setClassificationThreshold(classificationThreshold);
    }

    @Override
    public void setNmsThreshold(float nmsThreshold) {
        predictor.setNmsThreshold(nmsThreshold);
    }

    @Override
    public boolean isInit() {
        return predictor.isInit();
    }

    @Override
    public void release() {
        predictor.release();
    }
}
