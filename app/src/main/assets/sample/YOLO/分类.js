// 初始化YOLO分类器
let yolo = $yolo.init({
    type: 'onnx',
    modelPath: files.path('./best.onnx'),
    imageSize: 32,  //模型图片尺寸
    labels: ['0'],   //模型类列表，可选
    classificationThreshold: 0.000000001    //置信度，可选
});

if (!yolo) {
    toastLog('分类器初始化失败');
    exit();
}

let cf = $yolo.getInstance();
let imagePath = './test.png';
let img = images.read(imagePath);
let start = new Date()
let result = cf.predictClassification(img)
console.log(result);
cost = (new Date() - start)
toastLog('耗时' + cost + 'ms')
cf.release();
