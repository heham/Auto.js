package com.stardust.autojs.yolo;

import android.media.Image;
import com.stardust.autojs.runtime.api.Images;
import com.stardust.autojs.core.image.ImageWrapper;
import com.stardust.autojs.runtime.ScriptRuntime;
import com.stardust.autojs.yolo.onnx.domain.ClassificationResult;
import com.stardust.autojs.yolo.onnx.domain.DetectResult;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import java.util.Collections;
import java.util.List;

/**
 * YoloInstance是一个抽象类，定义了YOLO实例的基本行为和功能。
 */
public abstract class YoloInstance {

    public abstract YoloPredictor getPredictor();
    public abstract List<DetectResult> predictYolo(Mat image);
    public abstract List<ClassificationResult> predictClassification(Mat image);

    public void setConfThreshold(float confThreshold) {
        getPredictor().setConfThreshold(confThreshold);
    }

    public void setClassificationThreshold(float classificationThreshold) {
        getPredictor().setClassificationThreshold(classificationThreshold);
    }

    public void setNmsThreshold(float nmsThreshold) {
        getPredictor().setNmsThreshold(nmsThreshold);
    }

    public boolean isInit() {
        return getPredictor().isInit();
    }

    public void release() {
        getPredictor().release();
    }

    public List<DetectResult> captureAndPredict(ScriptRuntime runtime, Rect rect) {
        Images images = (Images) runtime.getImages();
        Image image = images.captureScreenRaw();
        if (image != null) {
            ImageWrapper imageWrapper = ImageWrapper.ofImageByMat(image, CvType.CV_8UC4);
            image.close();
            Mat mat = imageWrapper.getMat();
            if (rect != null) {
                Mat croppedImage = new Mat(mat, rect);
                mat.release();
                mat = croppedImage;
            }
            List<DetectResult> results = this.predictYolo(mat);
            mat.release();
            return results;
        }
        return Collections.emptyList();
    }

    public List<ClassificationResult> captureAndPredictClassification(ScriptRuntime runtime, Rect rect) {
        Images images = (Images) runtime.getImages();
        Image image = images.captureScreenRaw();
        if (image != null) {
            ImageWrapper imageWrapper = ImageWrapper.ofImageByMat(image, CvType.CV_8UC4);
            image.close();
            Mat mat = imageWrapper.getMat();
            if (rect != null) {
                Mat croppedImage = new Mat(mat, rect);
                mat.release();
                mat = croppedImage;
            }
            List<ClassificationResult> results = this.predictClassification(mat);
            mat.release();
            return results;
        }
        return Collections.emptyList();
    }
}
