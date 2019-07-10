# Multilabel-classification
Multilabel classification with mobilenet and inference in ncnn.  

Index
---
<!-- TOC -->

- [How to do multilabel classification.](#How to do multilabel classification.)  
    -[Change Label Map](##Change Label Map)  
    -[With scikit-learn tools](##With scikit-learn tools)  
    -[Tips](##Tips)  
    -[Change the Loss Function](##Change the Loss Function)  
    -[How to predict](##How to predict)  
    -[Sample Imbalance](##Sample Imbalance))  
    -[Spy](##Spy)  
-[How to Run the Model in Vs2017 with ncnn.](#How to Run the Model in Vs2017 with ncnn.)  
    -[Step](##Step)  
    -[Tips](##Tips)  
-[Model Compression](#Model Compression)  
    -[Quantization with tensorflow](##Quantization with tensorflow)  
    -[Quantization with ncnn](##Quantization with ncnn)  
 
<!-- /TOC -->
# How to do multilabel classification.
## Change Label Map
If we just want to do multiclass classification,the label map wil be:
> Cat:0  
> Dog:1  
> Other:2  

When we change label map:
> Cat:[1,0,0]  
> Dog:[0,1,0]  
> Other:[0,0,1]  
> Cat_Dog:[1,1,0]  
Beyond that,we comelete the operation like `one-hot-encode`.

## With scikit-learn tools
We can use `MultiLabelBinarizer` in scikit-learn.preprocess to transform text labels to one-hot like encode.
Usage:   
```
from sklearn.preprocessing import MultiLabelBinarizer
labels = [["cat","dog"]]
mlb = MultiLabelBinarizer
labels = mlb.fit_transform(labels)
```  
The result will be `[[1,1,0]]`.

## Tips
When we save the image and label list as a csv file, the type of the label will be string.
So we transform them with four line.
```
t_list = pd.read_csv("D:/project/ShuffleNet/csv/img1.2.csv")
t_list['label'] = t_list['label'].apply(lambda x: [int(x[1]), int(x[3]), int(x[5])])
t_list['label'] = t_list['label'].apply(lambda x: [float(x[0]), float(x[1]), float(x[2])])
train_label = t_list['label'].tolist()
``` 
## Change the Loss Function
When we complete a multilabel classification tast,we must use `sigmoid` loss function instead of `softmax`.
```
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=batch_labels)
```

We can also use label smoothing to improve model generalization ability.
```
label_smoothing = 0.1
batch_labels = (1.0-label_smoothing)*batch_labels+label_smoothing/N_CLASSES
```

## How to predict
We just compare the logits with threshold(0.5).
For one category,if the probability larger than the threshold,the label in that place is `1`.
```
model = MobileNetV1(x_4d, 3, is_training=False)
logit = model.output
pred = tf.nn.sigmoid(logit)
``` 
## Sample Imbalance
In fact,the number of the cat_dog images is far less than that of cat or dog.What we did is train a base model with all the datasets for 50 epochs.And forward the model with the input of cat、dog and other images,and then select 30000 of each category according to confidence.In the other words,we screen out the complex samples.We train another 50 epochs with new datasets using base model as pre-trained model.And finally,we get the results.
> Cat: 97.1% -> 95.8%  
> Dog: 97.4% -> 96.2%  
> Other: 97.31% -> 97.24%  
> Cat_Dog: 20% -> 80%  

## Spy
The number of the cat_dog images is very small.We need to spy more samples on the website.
Sometimes these images are damaged,so we must find out and delete them.
```
filename = tf.placeholder(tf.string, [], name='filename')
image_file = tf.read_file(filename)
image = tf.image.decode_jpeg(image_file)
sess=tf.Session()
for i in range(len(a)):
    print(a[i], '{}/{}'.format(i,len(a)))
    img=sess.run(image, feed_dict={filename:a[i]})
```  
`a` is a image path list,find the image and delete it.

# How to Run the Model in Vs2017 with ncnn.
ncnn[https://github.com/Tencent/ncnn] is a high-performance neural network inference computing framework optimized for mobile platforms.However it doesn't support tensorflow.So we must transfer to some bridge frame `coreml`,`onnx`.
## Step
> Step 1 Freeze the model to `.pb`
```
output_node_names = ["input_1","sigmoid_out"]
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)  
    output_graph_def = graph_util.convert_variables_to_constants(  
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names)
     with tf.gfile.GFile(output_graph, "wb") as f:  
          f.write(output_graph_def.SerializeToString())  
```
Get the output nodes ,restore the model and then freeze the graph.
> Step 2 Transform to coreml
```
tfcoreml.convert(tf_model_path="D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.pb",
                     mlmodel_path="D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.mlmodel",
                     output_feature_names=['sigmoid_out:0'],
                     input_name_shape_dict={'input_1:0': [1, 224, 224, 3]})
```

> Step 3 Transform to onnx
```
 model_coreml = load_spec('D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.mlmodel')
 model_onnx = convert_coreml(model_coreml, 7, name='ExampleModel')
 save_model(model_onnx, 'D:/Project/ShuffleNet/ncnn/save1/mobilenetv1.onnx')
 ```
> Step 4 Transform to ncnn
 You must build the ncnn and then use the tool `onnx2ncnn`
 ```
 onnx2ncnn.exe ../mobilenetv1.onnx ../mobilenetv1.param ../mobilenetv1.bin
 ```  
> Step 5 Run with ncnn
>> Load the model first;
```
ncnn::Net mobilenetv1;
mobilenetv1.load_param("D:/project/ShuffleNet/ncnn/save1/mobilenetv1.param");
mobilenetv1.load_model("D:/project/ShuffleNet/ncnn/save1/mobilenetv1.bin");
```
>> Input the image
We use image standardization method when we train.As a result,we must rewrite it with c plus plus.
```
cv::Mat mean_image, std_image;
cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
cv::cvtColor(image, image, CV_BGR2RGB);
cv::resize(image, image, Size(224, 224));
cv::meanStdDev(image, mean_image, std_image);
float mean1, std1, stda;
mean1 = (mean_image.at<double>(0, 0) + mean_image.at<double>(1, 0) + mean_image.at<double>(2, 0)) / 3;
std1 = (std_image.at<double>(0, 0) + std_image.at<double>(1, 0) + std_image.at<double>(2, 0)) / 3;
stda = max(std1, (1.0 / sqrt(224 * 224 * 3)));
```
After get the mean and standard deviation, we rewrite the ncnn mat input method.
```
namespace ncnn {
	static Mat from_rgb(const unsigned char* rgb, int w, int h, float mean, float std, Allocator* allocator)
	{
		Mat m(w, h, 3, 4u, allocator);
		if (m.empty())
			return m;

		float* ptr0 = m.channel(0);
		float* ptr1 = m.channel(1);
		float* ptr2 = m.channel(2);

		int size = w * h;
    
    ...
    
    for (; remain > 0; remain--)
		{
			*ptr0 = (rgb[0]-mean)/std;
			*ptr1 = (rgb[1]-mean)/std;
			*ptr2 = (rgb[2]-mean)/std;

			rgb += 3;
			ptr0++;
			ptr1++;
			ptr2++;
		}

		return m;
    }
}
```  
Finally,we can get the result.
```
ncnn::Mat in = ncnn::from_rgb(image.data, 224, 224, mean1, stda, 0);

ncnn::Extractor ex = mobilenetv1.create_extractor();
ex.input("input_1__0", in);

ncnn::Mat out;
ex.extract("sigmoid_out__0", out);
out = out.reshape(out.w * out.h * out.c);

cls_scores.resize(out.w);
for (int j = 0; j<out.w; j++)
{
	cls_scores[j] = out[j];
}
```  
## Tips 
* If you want to use MobileNetV2,please use relu instead of relu6.
* Coreml doesn't support ShuffleNet.
* You'd better use caffe instead of tensorflow if you want to use ncnn.

# Model Compression
## Quantization with tensorflow
You need to install bazel and build graph_transforms,then
```
bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=./mobilenetv1.pb \  
--out_graph=./mobilenetv1_int8.pb \  
--inputs=input_1 \  
--outputs=sigmoid_out \  
--transforms="quantize_weights"
```  
Surprisingly,the accuracy increase about 0.1%-0.2%.

## Quantization with ncnn
The mobilenetv1_int8.pb can't transform to coreml,so we can only use ncnn.
> ncnn2table
There are some problems in Windows,so some changes have been made instead of  command line work.
```
imagepath = "D:/trainimg1/";
parampath = "D:/project/ShuffleNet/ncnn/save1/mobilenetv1.param";
binpath = "D:/project/ShuffleNet/ncnn/save1/mobilenetv1.bin";
tablepath = "D:/project/ShuffleNet/ncnn/save1/mobilenetv1.table";
```  
> ncnn2int8
Then we build ncnn2int.exe and run
```
ncnn2int8.exe D:/project/ShuffleNet/ncnn/save1/mobilenetv1.param / 
D:/project/ShuffleNet/ncnn/save1/mobilenetv1.bin / 
D:/project/ShuffleNet/ncnn/save1/mobilenetv1_int8.param /
D:/project/ShuffleNet/ncnn/save1/mobilenetv1_int8.bin / 
D:/project/ShuffleNet/ncnn/save1/mobilenetv1.table 
```
>Run with int8_model
```
ncnn::Net mobilenetv1;
mobilenetv1.load_param("D:/project/ShuffleNet/ncnn/save1/mobilenetv1_int8.param");
mobilenetv1.load_model("D:/project/ShuffleNet/ncnn/save1/mobilenetv1_int8.bin");
```  


