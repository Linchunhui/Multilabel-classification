#include "stdafx.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "net.h"
#include <string>
#include <io.h>
#include <stdio.h>
#include <ctime>
#include "mat.h"
#include <limits.h>
#include <math.h>
#include <algorithm>
#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON
#include "platform.h"
using namespace std;
using namespace cv;
vector<cv::String> read_images_in_folder(cv::String pattern);
static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Net squeezenet;
	squeezenet.load_param("D:\\project1\\examples\\squeezenet_v1.1.param");
	squeezenet.load_model("D:\\project1\\examples\\squeezenet_v1.1.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 227, 227);

	const float mean_vals[3] = { 104.f, 117.f, 123.f };
	in.substract_mean_normalize(mean_vals, 0);

	ncnn::Extractor ex = squeezenet.create_extractor();

	ex.input("data", in);
	cout << "in.h:" << in.h << endl;
	cout << "in.w:" << in.w << endl;
	cout << "in.c:" << in.c << endl;

	ncnn::Mat out;
	ex.extract("conv10", out);
	cout << "out.h:" << out.h << endl;
	cout << "out.w:" << out.w << endl;
	cout << "out.c:" << out.c << endl;

	cls_scores.resize(out.w);
	for (int j = 0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
	}

	return 0;
}

static int detect_shufflenetv2(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Net shufflenetv2;
	shufflenetv2.load_param("D:/project/ShuffleNet/save/model1.param ");
	shufflenetv2.load_model("D:/project/ShuffleNet/save/model1.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

	const float mean_vals[3] = { 104.f, 117.f, 123.f };
	in.substract_mean_normalize(mean_vals, 0);

	ncnn::Extractor ex = shufflenetv2.create_extractor();

	ex.input("input_1:0", in);

	ncnn::Mat out;
	ex.extract("simgoid_out:0", out);
	cls_scores.resize(out.w);
	for (int j = 0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
	}

	return 0;
}

static int detect_mobilenetv2(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Net mobilenetv2;
	mobilenetv2.load_param("D:/project/ShuffleNet/ncnn/save/model_1.1.param ");
	mobilenetv2.load_model("D:/project/ShuffleNet/ncnn/save/model_1.1.bin");

	//ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);
	ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);

	//const float mean_vals[3] = { 104.f, 117.f, 123.f };
	//in.substract_mean_normalize(mean_vals, 0);

	//const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
	//in.substract_mean_normalize(0, norm_vals);
	ncnn::Extractor ex = mobilenetv2.create_extractor();

	ex.input("input_1__0", in);
	cout << "in.h:" << in.h << endl;
	cout << "in.w:" << in.w << endl;
	cout << "in.c:" << in.c << endl;

	ncnn::Mat out;
	ex.extract("sigmoid_out__0", out);
	cout<<" "<<endl;
	cout << "out.h:"<<out.h << endl;
	cout << "out.w:" << out.w << endl;
	cout << "out.c:" << out.c << endl;

	out = out.reshape(out.w * out.h * out.c);

	//{
//		ncnn::Layer* sigmoid = ncnn::create_layer("Sigmoid");
	//	ncnn::ParamDict pd;
		//sigmoid->load_param(pd);

		//sigmoid->forward_inplace(out);

//		delete sigmoid;
//	}
	//cout<<"data"<<out.a
	cls_scores.resize(out.w);
	for (int j = 0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
	}

	return 0;
}

static int detect_mobilenetv21(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
	ncnn::Net mobilenetv2;
	mobilenetv2.load_param("D:/project/ShuffleNet/ncnn/save/model_1.3.param ");
	mobilenetv2.load_model("D:/project/ShuffleNet/ncnn/save/model_1.3.bin");

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, 224, 224);

	//ncnn::Mat in;
	ncnn::Extractor ex = mobilenetv2.create_extractor();

	ex.input("input_1__0", in);
	//cout << "in.h:" << in.h << endl;
	//cout << "in.w:" << in.w << endl;
	//cout << "in.c:" << in.c << endl;

	ncnn::Mat out;
	ex.extract("sigmoid_out__0", out);
	//cout << " " << endl;
	//cout << "out.h:" << out.h << endl;
	//cout << "out.w:" << out.w << endl;
	//cout << "out.c:" << out.c << endl;

	out = out.reshape(out.w * out.h * out.c);
	//cout << "out.h:" << out.h << endl;
	//cout << "out.w:" << out.w << endl;
	//cout << "out.c:" << out.c << endl;

	cls_scores.resize(out.w);
	for (int j = 0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
	}

	return 0;
}
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

#if __ARM_NEON
		int nn = size >> 3;
		int remain = size - (nn << 3);
#else
		int remain = size;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
		for (; nn > 0; nn--)
		{
			uint8x8x3_t _rgb = vld3_u8(rgb);
			uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
			uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
			uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

			float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
			float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
			float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
			float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
			float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
			float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

			vst1q_f32(ptr0, _rlow);
			vst1q_f32(ptr0 + 4, _rhigh);
			vst1q_f32(ptr1, _glow);
			vst1q_f32(ptr1 + 4, _ghigh);
			vst1q_f32(ptr2, _blow);
			vst1q_f32(ptr2 + 4, _bhigh);

			rgb += 3 * 8;
			ptr0 += 8;
			ptr1 += 8;
			ptr2 += 8;
		}
#else
		if (nn > 0)
		{
			asm volatile(
				"0:                             \n"
				"pld        [%1, #256]          \n"
				"vld3.u8    {d0-d2}, [%1]!      \n"
				"vmovl.u8   q8, d0              \n"
				"vmovl.u8   q9, d1              \n"
				"vmovl.u8   q10, d2             \n"
				"vmovl.u16  q0, d16             \n"
				"vmovl.u16  q1, d17             \n"
				"vmovl.u16  q2, d18             \n"
				"vmovl.u16  q3, d19             \n"
				"vmovl.u16  q8, d20             \n"
				"vmovl.u16  q9, d21             \n"
				"vcvt.f32.u32   q0, q0          \n"
				"vcvt.f32.u32   q1, q1          \n"
				"vcvt.f32.u32   q2, q2          \n"
				"vcvt.f32.u32   q3, q3          \n"
				"vcvt.f32.u32   q8, q8          \n"
				"subs       %0, #1              \n"
				"vst1.f32   {d0-d3}, [%2 :128]! \n"
				"vcvt.f32.u32   q9, q9          \n"
				"vst1.f32   {d4-d7}, [%3 :128]! \n"
				"vst1.f32   {d16-d19}, [%4 :128]!\n"
				"bne        0b                  \n"
				: "=r"(nn),     // %0
				"=r"(rgb),    // %1
				"=r"(ptr0),   // %2
				"=r"(ptr1),   // %3
				"=r"(ptr2)    // %4
				: "0"(nn),
				"1"(rgb),
				"2"(ptr0),
				"3"(ptr1),
				"4"(ptr2)
				: "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10"
				);
		}
#endif // __aarch64__
#endif // __ARM_NEON
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
static int detect_mobilenetv22(std::string& img_path, std::vector<float>& cls_scores)
{
	ncnn::Net mobilenetv2;
	mobilenetv2.load_param("D:/project/ShuffleNet/ncnn/save/model_1.11.param");
	mobilenetv2.load_model("D:/project/ShuffleNet/ncnn/save/model_1.11.bin");
	cv::Mat mean_image, std_image;
	cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(image, image, CV_BGR2RGB);
	cv::resize(image, image, Size(224, 224));
	cv::meanStdDev(image, mean_image, std_image);
	float mean1, std1, stda;

	mean1 = (mean_image.at<double>(0, 0) + mean_image.at<double>(1, 0) + mean_image.at<double>(2, 0)) / 3;
	std1 = (std_image.at<double>(0, 0) + std_image.at<double>(1, 0) + std_image.at<double>(2, 0)) / 3;
	stda = max(std1, (1.0 / sqrt(224 * 224 * 3)));
	cout << "mean" << mean1 << endl;
	cout << "stda" << stda << endl;
	cout << " " << endl;
	ncnn::Mat in = ncnn::from_rgb(image.data, 224, 224, mean1,stda,0);

	ncnn::Extractor ex = mobilenetv2.create_extractor();
	ex.input("input_1__0", in);

	ncnn::Mat out;
	ex.extract("sigmoid_out__0", out);
	out = out.reshape(out.w * out.h * out.c);

	cls_scores.resize(out.w);
	for (int j = 0; j<out.w; j++)
	{
		cls_scores[j] = out[j];
	}

	return 0;
}

static int detect_mobilenetv1(std::string& img_path, std::vector<float>& cls_scores)
{
	ncnn::Net mobilenetv1;
	mobilenetv1.load_param("D:/project/ShuffleNet/ncnn/save1/mobilenetv1.6_int8.param");
	mobilenetv1.load_model("D:/project/ShuffleNet/ncnn/save1/mobilenetv1.6_int8.bin");
	cv::Mat mean_image, std_image;
	cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	cv::cvtColor(image, image, CV_BGR2RGB);
	cv::resize(image, image, Size(224, 224));
	cv::meanStdDev(image, mean_image, std_image);
	float mean1, std1, stda;

	mean1 = (mean_image.at<double>(0, 0) + mean_image.at<double>(1, 0) + mean_image.at<double>(2, 0)) / 3;
	std1 = (std_image.at<double>(0, 0) + std_image.at<double>(1, 0) + std_image.at<double>(2, 0)) / 3;
	stda = max(std1, (1.0 / sqrt(224 * 224 * 3)));

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

	return 0;
}
static int Image_Std(std::string& img_path,cv::Mat& img)
{
	cv::Mat mean_image, std_image;
	cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	//cout << image << endl;
	cv::cvtColor(image, image, CV_BGR2RGB);
	//cv::resize(image, image, Size(224, 224));
	cv::meanStdDev(image, mean_image, std_image);
	double mean1, std1,stda;
	//cout << image << endl;
	cout << image.rows << image.cols << image.dims << endl;
	mean1 = (mean_image.at<double>(0, 0)+ mean_image.at<double>(1, 0)+ mean_image.at<double>(2, 0))/3;
	std1= (std_image.at<double>(0, 0) + std_image.at<double>(1, 0) + std_image.at<double>(2, 0)) / 3;
	stda = max(std1, (1.0 / sqrt(224 * 224 * 3)));
	cout << "mean" << mean1 << endl;
	cout << "stda" << stda << endl;
	cout << " " << endl;
	//cv::Mat image1;
	//Mat image2;
	img.create(image.size(), CV_32FC3);
	//float array11[224][224][3];
	//image1 = (image - mean1)/stda;
	int height = image.rows;//获取RGB图像的行
	int width = image.cols;//获取RGB图像的列
	/*for (int i = 0; i < height; i++)
	{
		Vec3b* pixrow = image.ptr<Vec3b>(i);
		for (int j = 0; j < width; j++)
		{
			B = pixrow[j][0];
			G = pixrow[j][1];
			R = pixrow[j][2];
		}
	}*/
	for (int i = 0; i < image.rows; i++) {
		uchar* redrowptr = image.ptr<uchar>(i);
		float* redrowptr1 = img.ptr<float>(i);
		for (int j = 0; j < image.cols; j++) {
			float b = (redrowptr[3 * j + 0] - mean1) / stda;
			float g = (redrowptr[3 * j + 1] - mean1) / stda;
			float r = (redrowptr[3 * j + 2] - mean1) / stda;
			//cout << b << g << r <<endl;
			//array11[i][j][0] = b;
			//array11[i][j][1] = g;
			//array11[i][j][2] = r;
			redrowptr1[3 * j + 0] = b;
			redrowptr1[3 * j + 1] = g;
			redrowptr1[3 * j + 2] = r;
			//printf(" %f  %f  %f       \n", redrowptr1[3 * j + 0], redrowptr1[3 * j + 1], redrowptr1[3 * j + 2]);		
		}
	}
	/*for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			//获取RGB图像各通道像素值
			double b = image.at<Vec3b>(row, col)[0];//获取b通道像素值
			double g = image.at<Vec3b>(row, col)[1];//获取g通道像素值
			double r = image.at<Vec3b>(row, col)[2];//获取r通道像素值
											   //各通道取反
			double b1 = (b - mean1)/ stda;
			//cout << b1 << endl;
			double g1 = (g - mean1) / stda;
			double r1 = (r - mean1) / stda;
			//cout << b1 << endl;
			image1.ptr(row, col)[0] = b1;
			image1.ptr(row, col)[1] = g1;
			image1.prt<uchar>(row, col)[2] = r1;
		}
	}*/
	//cout << img << endl;
	//image1 = image1 / stda;
	//cv::cvtColor(img, img, CV_BGR2RGB);
	return 0;
}

static int print_topk(const std::vector<float>& cls_scores, int topk)
{
	// partial sort topk with index
	int size = cls_scores.size();
	std::vector< std::pair<float, int> > vec;
	vec.resize(size);
	for (int i = 0; i<size; i++)
	{
		vec[i] = std::make_pair(cls_scores[i], i);
	}
	//fprintf(stderr,vec.first)
	std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
		std::greater< std::pair<float, int> >());
	//fprintf(stderr, vec.first);
	// print topk and score
	for (int i = 0; i<topk; i++)
	{
		float score = vec[i].first;
		int index = vec[i].second;
		fprintf(stderr, "%d = %f\n", index, score);
		//fprintf(stderr, " = %f\n",  score);
	}

	return 0;
}

static int detect(std::string img_path, std::vector<float>& cls_scores) {
	
	cv::Mat img = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
	//Image_Std(img_path, img);

	//std::vector<float> cls_scores;
	//detect_squeezenet(img, cls_scores);
	//detect_shufflenetv2(img, cls_scores);
	detect_mobilenetv1(img_path, cls_scores);
	//detect_mobilenetv21(img, cls_scores);
	//cout << img_path << ":" << cls_scores[0] << "," << cls_scores[1] << "," << cls_scores[2] << endl;
	//if (cls_scores[0] > 0.5) {
	//	cout << "cat" << endl;
	//}
	return 0;
	//cout << img_path << ":" << cls_scores[0] << "," << cls_scores[1] << "," << cls_scores[2] << endl;
	//printf("Cat:%f Dog:%f Other:%f\n",cls_scores[0], cls_scores[1], cls_scores[2]);
	//cout << " " << endl;
	//return 0;
}
int main()
{
	cv::String pattern = "D:/data/cat1/*.jpg";
	vector<cv::String> fn = read_images_in_folder(pattern);
	size_t count = fn.size();
	//number of png files in images folder   
	Mat images;
	clock_t startTime, endTime;
	startTime = clock();
	float pred_cat, pred_dog, pred_other;
	int count_cat = 0;
	int count_dog = 0;
	int count_cat_dog = 0;
	int count_other = 0;
	for (size_t i = 0; i < count; i++)
	{
		std::vector<float> cls_scores;
		detect(fn[i],cls_scores);
		pred_cat = cls_scores[0];
		pred_dog = cls_scores[1];
		pred_other = cls_scores[2];
		cout <<i<<"/"<<count <<fn[i] << ":" << pred_cat << "," << pred_dog << "," << pred_other << endl;
		if ((pred_cat >= 0.4)&(pred_dog < 0.4)) {
			count_cat++;
		}
		if ((pred_cat < 0.4)&(pred_dog >= 0.4)) {
			count_dog++;
		}
		if ((pred_cat >= 0.4)&(pred_dog >= 0.4)) {
			count_cat_dog++;
		}
		if ((pred_cat < 0.4)&(pred_dog < 0.4)) {
			count_other++;
		}
		//images.push_back(imread(fn[i]));
		//imshow("img", imread(fn[i]));
		//waitKey(1000);
	}
	int c = count;
	endTime = clock();
	double time = (double)(endTime - startTime) / CLOCKS_PER_SEC;
	cout << "time" << time << endl;
	cout << "fps" << 1000 / time << endl;
	cout << "cat acc:" << count_cat<<","<<(double)count_cat/count<< endl;
	cout << "dog acc:" <<count_dog<<"," <<(double)count_dog / count << endl;
	cout << "cat_dog acc:" << count_cat_dog<<","<<(double)count_cat_dog / count << endl;
	cout << "other acc:" << (count_other / count) * 100 << endl;
	system("PAUSE");
	return 0;
}
vector<cv::String> read_images_in_folder(cv::String pattern)
{
	vector<cv::String> fn;
	glob(pattern, fn, false);
	vector<Mat> images;
	return fn;
}
