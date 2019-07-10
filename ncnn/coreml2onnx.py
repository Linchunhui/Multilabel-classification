from coremltools.models.utils import load_spec
from winmltools import convert_coreml
from winmltools.utils import save_model

def main():
    model_coreml = load_spec('D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.mlmodel')
    model_onnx = convert_coreml(model_coreml, 7, name='ExampleModel')
    save_model(model_onnx, 'D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.onnx')

if __name__=="__main__":
    main()