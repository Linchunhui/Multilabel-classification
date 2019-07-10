// BUG1989 is pleased to support the open source community by supporting ncnn available.
//
// author:BUG1989 (https://github.com/BUG1989/) Long-term support.
// author:JansonZhu (https://github.com/JansonZhu) Implemented the function of entropy calibration.
//
// Copyright (C) 2019 BUG1989. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include <stdafx.h>
#include <stdio.h>
#include <unistd.h>
//#include <getopt.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <stdlib.h>
#include <algorithm>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//#pragma comment(lib,"getopt.lib")
// ncnn public header
#include "platform.h"
#include "net.h"
#include "cpu.h"
#include "benchmark.h"

// ncnn private header
#include "layer/convolution.h"
#include "layer/convolutiondepthwise.h"
#include "layer/innerproduct.h"

using namespace std;

static ncnn::Option g_default_option;
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

// Get the filenames from direct path
int parse_images_dir(const char *base_path, std::vector<std::string>& file_path)
{
	DIR *dir;
	struct dirent *ptr;

	if ((dir = opendir(base_path)) == NULL)
	{
		perror("Open dir error...");
		exit(1);
	}

	while ((ptr = readdir(dir)) != NULL)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)    ///current dir OR parrent dir
		{
			continue;
		}

		std::string path = base_path;
		file_path.push_back(path + ptr->d_name);
		std::cout << "path:" << ptr->d_name << std::endl;
	}
	closedir(dir);

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
			*ptr0 = (rgb[0] - mean) / std;
			*ptr1 = (rgb[1] - mean) / std;
			*ptr2 = (rgb[2] - mean) / std;

			rgb += 3;
			ptr0++;
			ptr1++;
			ptr2++;
		}

		return m;
	}

}
class QuantNet : public ncnn::Net
{
public:
	int get_conv_names();
	int get_conv_bottom_blob_names();
	int get_conv_weight_blob_scales();

public:
	std::vector<std::string> conv_names;
	std::map<std::string, std::string> conv_bottom_blob_names;
	std::map<std::string, std::vector<float> > weight_scales;
};

int QuantNet::get_conv_names()
{
	for (size_t i = 0; i<layers.size(); i++)
	{
		ncnn::Layer* layer = layers[i];

		if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" || layer->type == "InnerProduct")
		{
			std::string name = layer->name;
			conv_names.push_back(name);
		}
	}

	return 0;
}

int QuantNet::get_conv_bottom_blob_names()
{
	// find conv bottom name or index
	for (size_t i = 0; i<layers.size(); i++)
	{
		ncnn::Layer* layer = layers[i];

		if (layer->type == "Convolution" || layer->type == "ConvolutionDepthWise" || layer->type == "InnerProduct")
		{
			std::string name = layer->name;
			std::string bottom_blob_name = blobs[layer->bottoms[0]].name;
			conv_bottom_blob_names[name] = bottom_blob_name;
		}
	}

	return 0;
}

int QuantNet::get_conv_weight_blob_scales()
{
	for (size_t i = 0; i<layers.size(); i++)
	{
		ncnn::Layer* layer = layers[i];

		if (layer->type == "Convolution")
		{
			std::string name = layer->name;
			const int weight_data_size_output = ((ncnn::Convolution*)layer)->weight_data_size / ((ncnn::Convolution*)layer)->num_output;
			std::vector<float> scales;

			// int8 winograd F43 needs weight data to use 6bit quantization
			bool quant_6bit = false;
			int kernel_w = ((ncnn::Convolution*)layer)->kernel_w;
			int kernel_h = ((ncnn::Convolution*)layer)->kernel_h;
			int dilation_w = ((ncnn::Convolution*)layer)->dilation_w;
			int dilation_h = ((ncnn::Convolution*)layer)->dilation_h;
			int stride_w = ((ncnn::Convolution*)layer)->stride_w;
			int stride_h = ((ncnn::Convolution*)layer)->stride_h;

			if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
				quant_6bit = true;

			for (int n = 0; n<((ncnn::Convolution*)layer)->num_output; n++)
			{
				const ncnn::Mat weight_data_n = ((ncnn::Convolution*)layer)->weight_data.range(weight_data_size_output * n, weight_data_size_output);
				const float *data_n = weight_data_n;
				float max_value = std::numeric_limits<float>::min();

				for (int i = 0; i < weight_data_size_output; i++)
					max_value = std::max(max_value, std::fabs(data_n[i]));

				if (quant_6bit)
					scales.push_back(31 / max_value);
				else
					scales.push_back(127 / max_value);
			}

			weight_scales[name] = scales;
		}

		if (layer->type == "ConvolutionDepthWise")
		{
			std::string name = layer->name;
			const int weight_data_size_output = ((ncnn::ConvolutionDepthWise*)layer)->weight_data_size / ((ncnn::ConvolutionDepthWise*)layer)->group;
			std::vector<float> scales;

			for (int n = 0; n<((ncnn::ConvolutionDepthWise*)layer)->group; n++)
			{
				const ncnn::Mat weight_data_n = ((ncnn::ConvolutionDepthWise*)layer)->weight_data.range(weight_data_size_output * n, weight_data_size_output);
				const float *data_n = weight_data_n;
				float max_value = std::numeric_limits<float>::min();

				for (int i = 0; i < weight_data_size_output; i++)
					max_value = std::max(max_value, std::fabs(data_n[i]));

				scales.push_back(127 / max_value);
			}

			weight_scales[name] = scales;
		}

		if (layer->type == "InnerProduct")
		{
			std::string name = layer->name;
			const int weight_data_size_output = ((ncnn::InnerProduct*)layer)->weight_data_size / ((ncnn::InnerProduct*)layer)->num_output;
			std::vector<float> scales;

			for (int n = 0; n<((ncnn::InnerProduct*)layer)->num_output; n++)
			{
				const ncnn::Mat weight_data_n = ((ncnn::InnerProduct*)layer)->weight_data.range(weight_data_size_output * n, weight_data_size_output);
				const float *data_n = weight_data_n;
				float max_value = std::numeric_limits<float>::min();

				for (int i = 0; i < weight_data_size_output; i++)
					max_value = std::max(max_value, std::fabs(data_n[i]));

				scales.push_back(127 / max_value);
			}

			weight_scales[name] = scales;
		}
	}

	return 0;
}

class QuantizeData
{
public:
	QuantizeData(std::string layer_name, int num);

	int initial_blob_max(ncnn::Mat data);
	int initial_histogram_interval();
	int initial_histogram_value();

	int normalize_histogram();
	int update_histogram(ncnn::Mat data);

	float compute_kl_divergence(const std::vector<float> &dist_a, const std::vector<float> &dist_b);
	int threshold_distribution(const std::vector<float> &distribution, const int target_bin = 128);
	float get_data_blob_scale();

public:
	std::string name;

	float max_value;
	int num_bins;
	float histogram_interval;
	std::vector<float> histogram;

	float threshold;
	int threshold_bin;
	float scale;
};

QuantizeData::QuantizeData(std::string layer_name, int num)
{
	name = layer_name;
	max_value = 0.0;
	num_bins = num;
	histogram_interval = 0.0;
	histogram.resize(num_bins);
	initial_histogram_value();
}

int QuantizeData::initial_blob_max(ncnn::Mat data)
{
	int channel_num = data.c;
	int size = data.w * data.h;

	for (int q = 0; q<channel_num; q++)
	{
		const float *data_n = data.channel(q);
		for (int i = 0; i<size; i++)
		{
			max_value = std::max(max_value, std::fabs(data_n[i]));
		}
	}

	return 0;
}

int QuantizeData::initial_histogram_interval()
{
	histogram_interval = max_value / num_bins;

	return 0;
}

int QuantizeData::initial_histogram_value()
{
	for (size_t i = 0; i<histogram.size(); i++)
	{
		histogram[i] = 0.00001;
	}

	return 0;
}

int QuantizeData::normalize_histogram()
{
	const int length = histogram.size();
	float sum = 0;

	for (int i = 0; i<length; i++)
		sum += histogram[i];

	for (int i = 0; i<length; i++)
		histogram[i] /= sum;

	return 0;
}

int QuantizeData::update_histogram(ncnn::Mat data)
{
	int channel_num = data.c;
	int size = data.w * data.h;

	for (int q = 0; q<channel_num; q++)
	{
		const float *data_n = data.channel(q);
		for (int i = 0; i<size; i++)
		{
			if (data_n[i] == 0)
				continue;

			int index = std::min(static_cast<int>(std::abs(data_n[i]) / histogram_interval), 2047);

			histogram[index]++;
		}
	}

	return 0;
}

float QuantizeData::compute_kl_divergence(const std::vector<float> &dist_a, const std::vector<float> &dist_b)
{
	const int length = dist_a.size();
	assert(dist_b.size() == length);
	float result = 0;

	for (int i = 0; i<length; i++)
	{
		if (dist_a[i] != 0)
		{
			if (dist_b[i] == 0)
			{
				result += 1;
			}
			else
			{
				result += dist_a[i] * log(dist_a[i] / dist_b[i]);
			}
		}
	}

	return result;
}

int QuantizeData::threshold_distribution(const std::vector<float> &distribution, const int target_bin)
{
	int target_threshold = target_bin;  //128
	float min_kl_divergence = 1000;
	const int length = distribution.size();

	std::vector<float> quantize_distribution(target_bin);

	float threshold_sum = 0;
	for (int threshold = target_bin; threshold<length; threshold++)
	{
		threshold_sum += distribution[threshold];
	}

	for (int threshold = target_bin; threshold<length; threshold++)
	{

		std::vector<float> t_distribution(distribution.begin(), distribution.begin() + threshold);

		t_distribution[threshold - 1] += threshold_sum;
		threshold_sum -= distribution[threshold];

		// get P
		fill(quantize_distribution.begin(), quantize_distribution.end(), 0);

		const float num_per_bin = static_cast<float>(threshold) / target_bin;

		for (int i = 0; i<target_bin; i++)
		{
			const float start = i * num_per_bin;
			const float end = start + num_per_bin;

			const int left_upper = ceil(start);
			if (left_upper > start)
			{
				const float left_scale = left_upper - start;
				quantize_distribution[i] += left_scale * distribution[left_upper - 1];
			}

			const int right_lower = floor(end);

			if (right_lower < end)
			{

				const float right_scale = end - right_lower;
				quantize_distribution[i] += right_scale * distribution[right_lower];
			}

			for (int j = left_upper; j<right_lower; j++)
			{
				quantize_distribution[i] += distribution[j];
			}
		}

		// get Q
		std::vector<float> expand_distribution(threshold, 0);

		for (int i = 0; i<target_bin; i++)
		{
			const float start = i * num_per_bin;
			const float end = start + num_per_bin;

			float count = 0;

			const int left_upper = ceil(start);
			float left_scale = 0;
			if (left_upper > start)
			{
				left_scale = left_upper - start;
				if (distribution[left_upper - 1] != 0)
				{
					count += left_scale;
				}
			}

			const int right_lower = floor(end);
			float right_scale = 0;
			if (right_lower < end)
			{
				right_scale = end - right_lower;
				if (distribution[right_lower] != 0)
				{
					count += right_scale;
				}
			}

			for (int j = left_upper; j<right_lower; j++)
			{
				if (distribution[j] != 0)
				{
					count++;
				}
			}

			const float expand_value = quantize_distribution[i] / count;

			if (left_upper > start)
			{
				if (distribution[left_upper - 1] != 0)
				{
					expand_distribution[left_upper - 1] += expand_value * left_scale;
				}
			}
			if (right_lower < end)
			{
				if (distribution[right_lower] != 0)
				{
					expand_distribution[right_lower] += expand_value * right_scale;
				}
			}
			for (int j = left_upper; j<right_lower; j++)
			{
				if (distribution[j] != 0)
				{
					expand_distribution[j] += expand_value;
				}
			}
		}

		// kl
		float kl_divergence = compute_kl_divergence(t_distribution, expand_distribution);

		// the best num of bin
		if (kl_divergence < min_kl_divergence)
		{
			min_kl_divergence = kl_divergence;
			target_threshold = threshold;
		}
	}

	return target_threshold;
}

float QuantizeData::get_data_blob_scale()
{
	normalize_histogram();
	threshold_bin = threshold_distribution(histogram);
	threshold = (threshold_bin + 0.5) * histogram_interval;
	scale = 127 / threshold;
	return scale;
}

struct PreParam
{
	float mean[3];
	float norm[3];
	int weith;
	int height;
};

//static int post_training_quantize(const std::vector<std::string> filenames, const char* param_path, const char* bin_path, const char* table_path, struct PreParam per_param)
static int post_training_quantize(const std::vector<std::string> filenames, const char* param_path, const char* bin_path, const char* table_path)
{
	int size = filenames.size();

	QuantNet net;

	net.load_param(param_path);
	std::cout <<"param" <<param_path << std::endl;
	net.load_model(bin_path);
	std::cout << "bin" << bin_path << std::endl;

	float mean_vals[3], norm_vals[3];
	int weith = 224;
	int height = 224;

	mean_vals[0] = 0.0f;
	mean_vals[1] = 0.0f;
	mean_vals[2] = 0.0f;

	norm_vals[0] = 1.f;
	norm_vals[1] = 1.f;
	norm_vals[2] = 1.f;

	g_blob_pool_allocator.clear();
	g_workspace_pool_allocator.clear();

	net.get_conv_names();
	net.get_conv_bottom_blob_names();
	net.get_conv_weight_blob_scales();

	FILE *fp = fopen(table_path, "w");
	// save quantization scale of weight 
	printf("====> Quantize the parameters.\n");
	for (size_t i = 0; i<net.conv_names.size(); i++)
	{
		//cout << "1" << endl;
		std::string layer_name = net.conv_names[i];
		std::string blob_name = net.conv_bottom_blob_names[layer_name];
		std::vector<float> weight_scale_n = net.weight_scales[layer_name];

		fprintf(fp, "%s_param_0 ", layer_name.c_str());
		for (size_t j = 0; j<weight_scale_n.size(); j++)
			fprintf(fp, "%f ", weight_scale_n[j]);
		fprintf(fp, "\n");
	}

	// initial quantization data
	std::vector<QuantizeData> quantize_datas;

	for (size_t i = 0; i<net.conv_names.size(); i++)
	{
		std::string layer_name = net.conv_names[i];

		QuantizeData quantize_data(layer_name, 2048);
		quantize_datas.push_back(quantize_data);
	}
	// step 1 count the max value
	printf("====> Quantize the activation.\n");
	printf("    ====> step 1 : find the max value.\n");

	for (size_t i = 0; i<filenames.size(); i++)
	{
		std::string img_name = filenames[i];

		if ((i + 1) % 100 == 0)
			fprintf(stderr, "          %d/%d\n", (int)(i + 1), (int)size);

		cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
		if (bgr.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
			return -1;
		}
		cv::Mat mean_image, std_image;
		//cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(bgr, bgr, CV_BGR2RGB);
		cv::resize(bgr, bgr, cv::Size(224, 224));
		cv::meanStdDev(bgr, mean_image, std_image);
		float mean1, std1, stda;

		mean1 = (mean_image.at<double>(0, 0) + mean_image.at<double>(1, 0) + mean_image.at<double>(2, 0)) / 3;
		std1 = (std_image.at<double>(0, 0) + std_image.at<double>(1, 0) + std_image.at<double>(2, 0)) / 3;
		//stda = max(std1, (1.0 / sqrt(224 * 224 * 3)));

		ncnn::Mat in = ncnn::from_rgb(bgr.data, 224, 224, mean1, stda, 0);

		//ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, weith, height);
		//in.substract_mean_normalize(mean_vals, norm_vals);

		ncnn::Extractor ex = net.create_extractor();
		ex.input("input_1__0", in);
		//cout << quantize_datas.size() << endl;
		for (size_t i = 0; i<net.conv_names.size(); i++)
		{
			std::string layer_name = net.conv_names[i];
			std::string blob_name = net.conv_bottom_blob_names[layer_name];

			ncnn::Mat out;
			ex.extract(blob_name.c_str(), out);
			for (size_t j = 0; j<quantize_datas.size(); j++)
			{	
				if (quantize_datas[j].name == layer_name)
				{
					quantize_datas[j].initial_blob_max(out);
					break;
				}
			}
		}
	}

	// step 2 histogram_interval
	printf("    ====> step 2 : generatue the histogram_interval.\n");
	for (size_t i = 0; i<net.conv_names.size(); i++)
	{
		std::string layer_name = net.conv_names[i];

		for (size_t j = 0; j<quantize_datas.size(); j++)
		{
			if (quantize_datas[j].name == layer_name)
			{
				quantize_datas[j].initial_histogram_interval();

				fprintf(stderr, "%-20s : max = %-15f interval = %-10f\n", quantize_datas[j].name.c_str(), quantize_datas[j].max_value, quantize_datas[j].histogram_interval);
				break;
			}
		}
	}

	// step 3 histogram
	printf("    ====> step 3 : generatue the histogram.\n");
	for (size_t i = 0; i<filenames.size(); i++)
	{
		std::string img_name = filenames[i];

		if ((i + 1) % 100 == 0)
			fprintf(stderr, "          %d/%d\n", (int)(i + 1), (int)size);

		cv::Mat bgr = cv::imread(img_name, CV_LOAD_IMAGE_COLOR);
		if (bgr.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", img_name.c_str());
			return -1;
		}
		cv::Mat mean_image, std_image;
		//cv::Mat image = cv::imread(img_path, CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(bgr, bgr, CV_BGR2RGB);
		cv::resize(bgr, bgr, cv::Size(224, 224));
		cv::meanStdDev(bgr, mean_image, std_image);
		float mean1, std1, stda;

		mean1 = (mean_image.at<double>(0, 0) + mean_image.at<double>(1, 0) + mean_image.at<double>(2, 0)) / 3;
		std1 = (std_image.at<double>(0, 0) + std_image.at<double>(1, 0) + std_image.at<double>(2, 0)) / 3;
		//stda = max(std1, (1.0 / sqrt(224 * 224 * 3)));

		ncnn::Mat in = ncnn::from_rgb(bgr.data, 224, 224, mean1, stda, 0);
		//ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, weith, height);
		//in.substract_mean_normalize(mean_vals, norm_vals);

		ncnn::Extractor ex = net.create_extractor();
		ex.input("input_1__0", in);

		for (size_t i = 0; i<net.conv_names.size(); i++)
		{
			std::string layer_name = net.conv_names[i];
			std::string blob_name = net.conv_bottom_blob_names[layer_name];

			ncnn::Mat out;
			ex.extract(blob_name.c_str(), out);

			for (size_t j = 0; j<quantize_datas.size(); j++)
			{
				if (quantize_datas[j].name == layer_name)
				{
					quantize_datas[j].update_histogram(out);
					break;
				}
			}
		}
	}

	// step4 kld
	printf("    ====> step 4 : using kld to find the best threshold value.\n");
	for (size_t i = 0; i<net.conv_names.size(); i++)
	{
		std::string layer_name = net.conv_names[i];
		std::string blob_name = net.conv_bottom_blob_names[layer_name];
		fprintf(stderr, "%-20s ", layer_name.c_str());

		for (size_t j = 0; j<quantize_datas.size(); j++)
		{
			if (quantize_datas[j].name == layer_name)
			{
				quantize_datas[j].get_data_blob_scale();
				fprintf(stderr, "bin : %-8d threshold : %-15f interval : %-10f scale : %-10f\n", \
					quantize_datas[j].threshold_bin, \
					quantize_datas[j].threshold, \
					quantize_datas[j].histogram_interval, \
					quantize_datas[j].scale);

				fprintf(fp, "%s %f\n", layer_name.c_str(), quantize_datas[j].scale);

				break;
			}
		}
	}

	fclose(fp);
	printf("====> Save the calibration table done.\n");

	return 0;
}

// usage
void showUsage()
{
	std::cout << "usage: ncnn2table [-h] [-p] [-b] [-o] [-m] [-n] [-s] [-t]" << std::endl;
	std::cout << " -h, --help       show this help message and exit" << std::endl;
	std::cout << " -p, --param      path to ncnn.param file" << std::endl;
	std::cout << " -b, --bin        path to ncnn.bin file" << std::endl;
	std::cout << " -o, --output     path to output calibration tbale file" << std::endl;
	std::cout << " -m, --mean       value of mean" << std::endl;
	std::cout << " -n, --norm       value of normalize(scale value,defualt is 1)" << std::endl;
	std::cout << " -s, --size       the size of input image(using the resize the original image,default is w=224,h=224)" << std::endl;
	std::cout << " -t, --thread     number of threads(defalut is 1)" << std::endl;
	std::cout << "example: ./ncnn2table --param squeezenet-fp32.param --bin squeezenet-fp32.bin --images images/ --output squeezenet.table --mean 104,117,123 --norm 1,1,1 --size 227,227 --thread 2" << std::endl;
}

// string.split('x')
std::vector<std::string> split(const std::string &str, const std::string &pattern)
{
	//const char* convert to char*
	char * strc = new char[strlen(str.c_str()) + 1];
	strcpy(strc, str.c_str());
	std::vector<std::string> resultVec;
	char* tmpStr = strtok(strc, pattern.c_str());
	while (tmpStr != NULL)
	{
		resultVec.push_back(std::string(tmpStr));
		tmpStr = strtok(NULL, pattern.c_str());
	}

	delete[] strc;

	return resultVec;
}

int main()
{
	std::cout << "--- ncnn post training quantization tool --- " << __TIME__ << " " << __DATE__ << std::endl;

	char* imagepath = NULL;
	char* parampath = NULL;
	char* binpath = NULL;
	char* tablepath = NULL;
	int num_threads = 1;

	/*   struct PreParam pre_param = {
	.mean = {104.f, 117.f, 103.f},
	.norm = {1.f, 1.f, 1.f},
	.weith = 224,
	.height =224
	};*/

	int c;
	imagepath = "D:/trainimg1/";
	parampath = "D:/project/ShuffleNet/ncnn/save1/mobilenetv1.param";
	binpath = "D:/project/ShuffleNet/ncnn/save1/mobilenetv1.bin";
	tablepath = "D:/project/ShuffleNet/ncnn/save1/mobilenetv1.7.table";

	/*while (1)
	{
	int option_index = 0;
	static struct option long_options[] =
	{
	{ "param",   required_argument, 0,  'p' },
	{ "bin",     required_argument, 0,  'b' },
	{ "images",  required_argument, 0,  'i' },
	{ "output",  required_argument, 0,  'o' },
	{ "mean",    required_argument, 0,  'm' },
	{ "norm",    required_argument, 0,  'n' },
	{ "size",    required_argument, 0,  's' },
	{ "thread",  required_argument, 0,  't' },
	{ "help",    no_argument,       0,  'h' },
	{ 0,         0,                 0,  0 }
	};

	c = getopt_long(argc, argv, "p:b:i:o:m:n:s:t:h", long_options, &option_index);
	if (c == -1)
	break;

	switch (c)
	{
	case 'p':
	printf("param = '%s'\n", optarg);
	parampath = optarg;
	break;

	case 'b':
	printf("bin = '%s'\n", optarg);
	binpath = optarg;
	break;

	case 'i':
	printf("images = '%s'\n", optarg);
	imagepath = optarg;
	break;

	case 'o':
	printf("output = '%s'\n", optarg);
	tablepath = optarg;
	break;

	/* case 'm':
	{
	printf("mean = '%s'\n", optarg);
	std::string temp(optarg);
	std::vector<std::string> array = split(temp, ",");
	pre_param.mean[0] = atof(array[0].c_str());
	pre_param.mean[1] = atof(array[1].c_str());
	pre_param.mean[2] = atof(array[2].c_str());
	}
	break;

	case 'n':
	{
	printf("norm = '%s'\n", optarg);
	std::string temp(optarg);
	std::vector<std::string> array = split(temp, ",");
	pre_param.norm[0] = atof(array[0].c_str());
	pre_param.norm[1] = atof(array[1].c_str());
	pre_param.norm[2] = atof(array[2].c_str());
	}
	break;

	case 's':
	{
	printf("size = '%s'\n", optarg);
	std::string temp(optarg);
	std::vector<std::string> array = split(temp, ",");
	pre_param.weith = atoi(array[0].c_str());
	pre_param.height = atoi(array[1].c_str());
	}
	break;

	case 't':
	printf("thread = '%s'\n", optarg);
	num_threads = atoi(optarg);
	break;

	case 'h':
	case '?':
	showUsage();
	return 0;

	default:
	showUsage();
	}
	}*/

	// check the input param
	if (imagepath == NULL || parampath == NULL || binpath == NULL || tablepath == NULL)
	{
		fprintf(stderr, "someone path maybe empty,please check it and try again.\n");
		return 0;
	}

	g_blob_pool_allocator.set_size_compare_ratio(0.0f);
	g_workspace_pool_allocator.set_size_compare_ratio(0.5f);

	// default option
	g_default_option.lightmode = true;
	g_default_option.num_threads = num_threads;
	g_default_option.blob_allocator = &g_blob_pool_allocator;
	g_default_option.workspace_allocator = &g_workspace_pool_allocator;

	g_default_option.use_winograd_convolution = true;
	g_default_option.use_sgemm_convolution = true;
	g_default_option.use_int8_inference = true;
	g_default_option.use_fp16_packed = true;
	g_default_option.use_fp16_storage = true;
	g_default_option.use_fp16_arithmetic = true;
	g_default_option.use_int8_storage = true;
	g_default_option.use_int8_arithmetic = true;

	ncnn::set_cpu_powersave(2);
	ncnn::set_omp_dynamic(0);
	ncnn::set_omp_num_threads(num_threads);

	std::vector<std::string> filenames;

	// parse the image file.
	parse_images_dir(imagepath, filenames);

	// get the calibration table file, and save it.
	int ret = post_training_quantize(filenames, parampath, binpath, tablepath);
	if (!ret)
		fprintf(stderr, "\nNCNN Int8 Calibration table create success, best wish for your INT8 inference has a low accuracy loss...\\(^¨Œ^)/...233...\n");
	system("pause");
	return 0;
}
