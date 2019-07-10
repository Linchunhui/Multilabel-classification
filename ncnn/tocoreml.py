import tfcoreml
import sys
sys.path = "D:/Project/ShuffleNet/save"
def main():
    tfcoreml.convert(tf_model_path="D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.pb",
                     mlmodel_path="D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.mlmodel",
                     output_feature_names=['sigmoid_out:0'],
                     input_name_shape_dict={'input_1:0': [1, 224, 224, 3]})
if __name__=="__main__":
    main()