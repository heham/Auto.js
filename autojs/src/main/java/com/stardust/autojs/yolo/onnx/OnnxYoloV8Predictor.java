package com.stardust.autojs.yolo.onnx;

import android.os.Build;
import android.util.Log;

import com.google.gson.Gson;
import com.stardust.autojs.yolo.YoloPredictor;
import com.stardust.autojs.yolo.onnx.domain.DetectResult;
import com.stardust.autojs.yolo.onnx.domain.Detection;
import com.stardust.autojs.yolo.onnx.domain.ClassificationResult;
import com.stardust.autojs.yolo.onnx.util.Letterbox;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.providers.NNAPIFlags;
import androidx.annotation.RequiresApi;

/**
 * @author TonyJiangWJ
 * @since 2023/8/20
 * transfer from https://gitee.com/agricultureiot/yolo-onnx-java
 */
@RequiresApi(api = Build.VERSION_CODES.N)
public class OnnxYoloV8Predictor extends YoloPredictor {
    private static final String TAG = "YoloV8Predictor";
    private static final Pattern IMG_SIZE_PATTERN = Pattern.compile("\\[(\\d+), \\d+]");
    private static final Pattern LABEL_PATTERN = Pattern.compile("'([^']*)'");

    private final String modelPath;

    private boolean tryNpu;
    private Size shapeSize = new Size(640, 640);
    private Letterbox letterbox;

    private List<String> apiFlags = Arrays.asList("CPU_DISABLED");

    public OnnxYoloV8Predictor(String modelPath) {
        this.modelPath = modelPath;
        init = true;
    }

    public OnnxYoloV8Predictor(String modelPath, float confThreshold, float nmsThreshold) {
        this.modelPath = modelPath;
        this.confThreshold = confThreshold;
        this.nmsThreshold = nmsThreshold;
        init = true;
    }

    public void setShapeSize(double width, double height) {
        this.shapeSize = new Size(width, height);
    }

    public void setTryNpu(boolean tryNpu) {
        this.tryNpu = tryNpu;
    }

    public void setApiFlags(List<String> apiFlags) {
        this.apiFlags = apiFlags;
    }

    private OrtSession session;
    private OrtEnvironment environment;

    private void prepareSession() throws OrtException {
        if (environment != null) {
            return;
        }
        // 加载ONNX模型
        environment = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        addNNApiProvider(sessionOptions);

        session = environment.createSession(modelPath, sessionOptions);
        // 输出基本信息
        for (String inputName : session.getInputInfo().keySet()) {
            try {
                System.out.println("input name = " + inputName);
                System.out.println(session.getInputInfo().get(inputName).getInfo().toString());
            } catch (OrtException e) {
                throw new RuntimeException(e);
            }
        }
        // 如果入参labels无效或未定义，使用模型内置labels
        if (labels == null || labels.size() == 0) {
            labels = initLabels(session);
        }
        initShapeSize(session);
    }

    private List<String> initLabels(OrtSession session) throws OrtException {
        String metaStr = session.getMetadata().getCustomMetadata().get("names");
        if (metaStr == null) {
            Log.d(TAG, "initLabels: 读取names失败 无法自动修正labels");
            return Collections.emptyList();
        }
        List<String> labelList = new ArrayList<>();
        
        Matcher matcher = LABEL_PATTERN.matcher(metaStr);
        while (matcher.find()) {
            labelList.add(matcher.group(1));
        }
        
        return labelList;
    }

    private void initShapeSize(OrtSession session) throws OrtException {
        String metaStr = session.getMetadata().getCustomMetadata().get("imgsz");
        Log.d(TAG, "initShapeSize: " + metaStr);
        if (metaStr == null) {
            Log.d(TAG, "initShapeSize: 读取imgsz失败 无法自动修正输入大小");
            return;
        }
        Matcher matcher = IMG_SIZE_PATTERN.matcher(metaStr);
        if (matcher.find()) {
            String shapeSizeStr = matcher.group(1);
            if (shapeSizeStr != null) {
                this.shapeSize = new Size(Double.parseDouble(shapeSizeStr), Double.parseDouble(shapeSizeStr));
                Log.d(TAG, "set shape size: " + shapeSizeStr);
            }
        } else {
            Log.d(TAG, "initShapeSize: 读取imgsz格式异常 无法自动修正输入大小");
        }
    }

    private void addNNApiProvider(OrtSession.SessionOptions sessionOptions) {
        if (!tryNpu) {
            return;
        }
        try {
            List<NNAPIFlags> flags = new ArrayList<>();
            if (apiFlags.contains("USE_FP16")) {
                flags.add(NNAPIFlags.USE_FP16);
            }
            if (apiFlags.contains("USE_NCHW")) {
                flags.add(NNAPIFlags.USE_NCHW);
            }
            if (apiFlags.contains("CPU_ONLY")) {
                flags.add(NNAPIFlags.CPU_ONLY);
            }
            if (apiFlags.contains("CPU_DISABLED")) {
                flags.add(NNAPIFlags.CPU_DISABLED);
            }
            Log.d(TAG, "addNNApiProvider: 当前启用nnapiFlags:" + new Gson().toJson(apiFlags));
            sessionOptions.addNnapi(EnumSet.copyOf(flags));
            Log.d(TAG, "prepareSession: 启用nnapi成功");
        } catch (Exception e) {
            Log.e(TAG, "prepareSession: 无法启用nnapi");
        }
    }

    private HashMap<String, OnnxTensor> preprocessImage(Mat img) throws OrtException {
        Mat image = img.clone();
        if (image.channels() == 4) {
            Imgproc.cvtColor(image, image, Imgproc.COLOR_RGBA2RGB);
        }
        Log.d(TAG, "preprocessImage: image's channels: " + image.channels());
        
        letterbox = new Letterbox();
        letterbox.setNewShape(this.shapeSize);
        image = letterbox.letterbox(image);

        int rows = letterbox.getHeight();
        int cols = letterbox.getWidth();
        int channels = image.channels();
        
        Mat convertedImage = new Mat();
        image.convertTo(convertedImage, CvType.CV_32FC3, 1.0 / 255.0);

        float[] pixelData = new float[rows * cols * channels];
        convertedImage.get(0, 0, pixelData);

        float[] pixels = new float[channels * rows * cols];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < rows; h++) {
                for (int w = 0; w < cols; w++) {
                    int index = c * rows * cols + h * cols + w;
                    int srcIndex = (h * cols + w) * channels + c;
                    pixels[index] = pixelData[srcIndex];
                }
            }
        }
        
        image.release();
        convertedImage.release();
        
        long[] shape = {1L, (long) channels, (long) rows, (long) cols};
        OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), shape);
        
        HashMap<String, OnnxTensor> inputMap = new HashMap<>();
        String inputName = session.getInputInfo().keySet().iterator().next();
        inputMap.put(inputName, tensor);
        
        return inputMap;
    }

    private List<Detection> postProcessOutput(Result output) throws OrtException {
        float[][][] outputArray = (float[][][]) output.get(0).getValue();
        float[][] outputData = outputArray[0];

        outputData = transposeMatrix(outputData);
        Map<Integer, List<float[]>> class2Bbox = new HashMap<>();

        for (float[] bbox : outputData) {
            int label = argmax(bbox, 4);
            float conf = bbox[label + 4];
            if (conf < confThreshold) {
                continue;
            }

            bbox[4] = conf;
            xywh2xyxy(bbox);

            if (bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) {
                continue;
            }

            class2Bbox.computeIfAbsent(label, k -> new ArrayList<>()).add(bbox);
        }

        List<Detection> detections = new ArrayList<>();
        for (Map.Entry<Integer, List<float[]>> entry : class2Bbox.entrySet()) {
            int label = entry.getKey();
            List<float[]> bboxes = entry.getValue();
            bboxes = nonMaxSuppression(bboxes, nmsThreshold);
            for (float[] bbox : bboxes) {
                String labelString = (labels != null && labels.size() > label) ? labels.get(label) : String.valueOf(label);
                detections.add(new Detection(labelString, label, Arrays.copyOfRange(bbox, 0, 4), bbox[4]));
            }
        }
        return detections;
    }

    public List<DetectResult> predictYolo(String imagePath) throws OrtException {
        return predictYolo(Imgcodecs.imread(imagePath));
    }

    public List<DetectResult> predictYolo(Mat image) throws OrtException {
        prepareSession();
        long start_time = System.currentTimeMillis();
        Map<String, OnnxTensor> inputMap = preprocessImage(image);
        try (Result output = session.run(inputMap)) {
            Log.d(TAG, "predictYolo: onnx run cost " + (System.currentTimeMillis() - start_time) + "ms");
            List<Detection> detections = postProcessOutput(output);
            Log.d(TAG, String.format("onnx predict cost: %d ms", (System.currentTimeMillis() - start_time)));
            List<DetectResult> results = new ArrayList<>();
            for (Detection detection : detections) {
                results.add(new DetectResult(detection, letterbox));
            }
            return results;
        } finally {
            for (OnnxTensor tensor : inputMap.values()) {
                tensor.close();
            }
        }
    }

    public static void xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];

        bbox[0] = x - w * 0.5f;
        bbox[1] = y - h * 0.5f;
        bbox[2] = x + w * 0.5f;
        bbox[3] = y + h * 0.5f;
    }

    public static float[][] transposeMatrix(float[][] m) {
        float[][] temp = new float[m[0].length][m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                temp[j][i] = m[i][j];
            }
        }
        return temp;
    }

    public static List<float[]> nonMaxSuppression(List<float[]> bboxes, float iouThreshold) {
        long start = System.currentTimeMillis();
        List<float[]> bestBboxes = new ArrayList<>();

        bboxes.sort(Comparator.comparing(a -> a[4]));

        while (!bboxes.isEmpty()) {
            float[] bestBbox = bboxes.remove(bboxes.size() - 1);
            bestBboxes.add(bestBbox);
            bboxes.removeIf(bbox -> computeIOU(bbox, bestBbox) >= iouThreshold);
        }
        Log.d(TAG, "nonMaxSuppression: cost " + (System.currentTimeMillis() - start) + "ms");
        return bestBboxes;
    }

    public static float computeIOU(float[] box1, float[] box2) {
        float area1 = (box1[2] - box1[0]) * (box1[3] - box1[1]);
        float area2 = (box2[2] - box2[0]) * (box2[3] - box2[1]);

        float left = Math.max(box1[0], box2[0]);
        float top = Math.max(box1[1], box2[1]);
        float right = Math.min(box1[2], box2[2]);
        float bottom = Math.min(box1[3], box2[3]);

        float width = Math.max(right - left, 0);
        float height = Math.max(bottom - top, 0);

        float interArea = width * height;
        float unionArea = area1 + area2 - interArea;

        return Math.max(interArea / unionArea, 1e-8f);
    }

    public static int argmax(float[] a, int start) {
        float max = -Float.MAX_VALUE;
        int arg = -1;
        for (int i = start; i < a.length; i++) {
            if (a[i] > max) {
                max = a[i];
                arg = i - start;
            }
        }
        return arg;
    }

    // ------------------ 分类相关 ------------------
    @Override
    public List<ClassificationResult> predictClassification(Mat image) throws OrtException {
        prepareSession();
        Map<String, OnnxTensor> inputMap = preprocessForClassification(image);
        try (Result output = session.run(inputMap)) {
            return postProcessClassificationOutput(output);
        } finally {
            for (OnnxTensor tensor : inputMap.values()) {
                tensor.close();
            }
        }
    }

    private List<ClassificationResult> postProcessClassificationOutput(Result output) throws OrtException {
    List<ClassificationResult> results = new ArrayList<>();
    try {
        Object raw = output.get(0).getValue();

        float[] probabilities;
        if (raw instanceof float[][]) {
            probabilities = ((float[][]) raw)[0];
        } else if (raw instanceof float[][][]) {
            probabilities = ((float[][][]) raw)[0][0];
        } else {
            Log.e(TAG, "Unsupported output type: " + raw.getClass());
            return Collections.emptyList();
        }

        // 判断是否已经是概率
        float sum = 0f;
        for (float v : probabilities) sum += v;
        if (Math.abs(sum - 1.0f) > 1e-3) {
            // 不是概率，做 softmax
            probabilities = softmax(probabilities);
        } else {
            Log.d(TAG, "Output already normalized, skip softmax");
        }

        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] >= classificationThreshold) {
                String labelName = (labels != null && labels.size() > i) ? labels.get(i) : "Class_" + i;
                results.add(new ClassificationResult(labelName, i, probabilities[i]));
            }
        }

        results.sort((a, b) -> Float.compare(b.getConfidence(), a.getConfidence()));
        return results;

    } catch (Exception e) {
        Log.e(TAG, "Error processing classification output", e);
        return Collections.emptyList();
    }
}



    private float[] softmax(float[] logits) {
        float maxLogit = logits[0];
        for (float value : logits) {
            if (value > maxLogit) maxLogit = value;
        }

        float sum = 0.0f;
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += exps[i];
        }

        for (int i = 0; i < exps.length; i++) {
            exps[i] /= sum;
        }
        return exps;
    }

    private Map<String, OnnxTensor> preprocessForClassification(Mat img) throws OrtException {
        Mat image = img.clone();
        if (image.channels() == 4) {
            Imgproc.cvtColor(image, image, Imgproc.COLOR_RGBA2RGB);
        }

        // 使用模型指定的输入尺寸
        Imgproc.resize(image, image, new Size(shapeSize.width, shapeSize.height));
        Mat convertedImage = new Mat();
        image.convertTo(convertedImage, CvType.CV_32FC3, 1.0 / 255.0);

        int rows = convertedImage.rows();
        int cols = convertedImage.cols();
        int channels = convertedImage.channels();

        float[] pixelData = new float[rows * cols * channels];
        convertedImage.get(0, 0, pixelData);

        float[] pixels = new float[channels * rows * cols];
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < rows; h++) {
                for (int w = 0; w < cols; w++) {
                    int index = c * rows * cols + h * cols + w;
                    int srcIndex = (h * cols + w) * channels + c;
                    pixels[index] = pixelData[srcIndex];
                }
            }
        }

        long[] shape = {1L, (long) channels, (long) rows, (long) cols};
        OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(pixels), shape);

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        String inputName = session.getInputInfo().keySet().iterator().next();
        inputMap.put(inputName, tensor);

        image.release();
        convertedImage.release();

        return inputMap;
    }

    @Override
    public void release() {
        this.init = false;
        if (session != null) {
            try {
                session.close();
            } catch (OrtException e) {
                Log.e(TAG, "close session failed", e);
            }
            session = null;
        }
        if (environment != null) {
            environment.close();
            environment = null;
        }
    }
}
