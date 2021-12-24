//
// Created by xiongzhuang on 2021/10/8.
//
#include <utility>
#include "timer.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include "LightTrack.h"

void print_rknn(float *data, const char* file_name)
{
    std::ofstream OutFile(file_name);

    for (int i=0; i<4*18*18; i++)
    {
        std::stringstream ss;
        std::string s;
        ss << data[i];
        s = ss.str();
        OutFile << s;
        OutFile << "\n";
    }
    OutFile.close();
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }


    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, const cv::Point2f& target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2;
}


LightTrack::LightTrack(const std::string& model_init, const std::string& model_backbone, const std::string& model_neck_head)
{

    score_size = int(round(this->instance_size / this->total_stride));

    int model_len=0;

    // net init
    init_model_data = LightTrack::load_model(model_init.c_str(), &model_len);
    rknn_init(&(this->net_init), init_model_data, model_len, 0);

    // net backbone
    backbone_model_data = LightTrack::load_model(model_backbone.c_str(), &model_len);
    rknn_init(&(this->net_backbone), backbone_model_data, model_len, 0);

    // net neck_head
    neck_head_model_data = LightTrack::load_model(model_neck_head.c_str(), &model_len);
    rknn_init(&(this->net_neck_head), neck_head_model_data, model_len, 0);

}

LightTrack::~LightTrack()
{
    rknn_outputs_release(net_init, 1, zf);

    rknn_destroy(net_init);
    rknn_destroy(net_backbone);
    rknn_destroy(net_neck_head);

    if (init_model_data)
    {
        free(init_model_data);
    }

    if (backbone_model_data)
    {
        free(backbone_model_data);
    }

    if (neck_head_model_data)
    {
        free(neck_head_model_data);
    }
}

void LightTrack::init(const cv::Mat& img, cv::Point target_pos_, cv::Point2f target_sz_)
{
    this->target_pos = std::move(target_pos_);
    this->target_sz = std::move(target_sz_);

    std::cout << "init target pos: " << target_pos << std::endl;
    std::cout << "init target_sz: " << target_sz << std::endl;

    this->grids();

    // 对模板图像而言：在第一帧以s_z为边长，以目标中心为中心点，截取图像补丁（如果超出第一帧的尺寸，用均值填充）。之后将其resize为127x127x3.成为模板图像
    // context = 1/2 * (w+h) = 2*pad
    float wc_z = target_sz.x + this->context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + this->context_amount * (target_sz.x + target_sz.y);
    // z_crop size = sqrt((w+2p)*(h+2p))
    float s_z = round(sqrt(wc_z * hc_z));   // orignal size

    cv::Scalar avg_chans = cv::mean(img);
    cv::Mat z_crop;

    z_crop  = get_subwindow_tracking(img, target_pos, this->exemplar_size, int(s_z));

    // z_crop = cv::imread("000034-baby0_127.jpg");

    // Set Input Data
    cv::Mat rgb;
    cv::cvtColor(z_crop, rgb, cv::COLOR_BGR2RGB);
    rknn_input rknn_img[1];
    memset(rknn_img, 0, sizeof(rknn_img));
    rknn_img[0].index = 0;
    rknn_img[0].type = RKNN_TENSOR_UINT8;
    rknn_img[0].size = rgb.cols*rgb.rows*rgb.channels();
    rknn_img[0].fmt = RKNN_TENSOR_NHWC;
    rknn_img[0].buf = rgb.data;
    rknn_inputs_set(net_init, 1, rknn_img);

    // Run
    rknn_run(net_init, nullptr);

    // Get Output
    memset(zf, 0, sizeof(zf));
    for (auto & i : zf) {
        i.want_float = 1;
        i.is_prealloc = 0;
    }
    rknn_outputs_get(net_init, 1, zf, nullptr);

    float* zf_data = (float*)zf[0].buf;
    // print_rknn(zf_data, "zf_rknn.txt");

    std::vector<float> hanning(this->score_size,0);  // 18

    this->window.resize(this->score_size * this->score_size, 0);
    for (int i = 0; i < this->score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * PI * float(i) / float(this->score_size - 1));
        hanning[i] = w;
    }
    for (int i = 0; i < this->score_size; i++)
    {

        for (int j = 0; j < this->score_size; j++)
        {
            this->window[i*this->score_size+j] = hanning[i] * hanning[j];
        }
    }

}

void LightTrack::update(const cv::Mat &x_crop, float scale_z)
{
    time_checker time2{}, time3{}, time4{}, time5{};

    /* net backbone */
    time2.start();
    // Set Input Data
    rknn_input rknn_img[1];
    memset(rknn_img, 0, sizeof(rknn_img));
    cv::cvtColor(x_crop, x_crop, cv::COLOR_BGR2RGB);
    rknn_img[0].index = 0;
    rknn_img[0].type = RKNN_TENSOR_UINT8;
    rknn_img[0].size = x_crop.cols*x_crop.rows*x_crop.channels();
    rknn_img[0].fmt = RKNN_TENSOR_NHWC;
    rknn_img[0].buf = x_crop.data;
    rknn_inputs_set(net_backbone, 1, rknn_img);
    time2.stop();
    time2.show_distance("Update stage ---- input seting cost time");

    time3.start();
    // Run
    rknn_run(net_backbone, nullptr);
    // Get Output
    rknn_output xf[1];
    memset(xf, 0, sizeof(xf));
    for (auto & i : xf) {
        i.want_float = 1;
        i.is_prealloc = 0;
    }
    rknn_outputs_get(net_backbone, 1, xf, nullptr);

    float* xf_data = (float*)xf[0].buf;
    // print_rknn(xf_data, "xf_rknn_cpp.txt");
    time3.stop();
    time3.show_distance("Update stage ---- output xf extracting cost time");

    /* net neck head */
    time4.start();
    rknn_input zf_xf[2];
    memset(zf_xf, 0, sizeof(zf_xf));
    zf_xf[0].index = 0;
    zf_xf[0].type = RKNN_TENSOR_FLOAT32;
    zf_xf[0].size = zf[0].size;
    zf_xf[0].fmt = RKNN_TENSOR_NCHW;
    zf_xf[0].buf = zf[0].buf;
    zf_xf[0].pass_through = 0;   // 这里必须为0，当为1时设置的fmt将不起作用，就会时默认的NHWC，注意rknn的输入都是NHWC格式，输出都是NCHW格式，所以这里的xf和zf都是NCHW格式
    zf_xf[1].index = 1;
    zf_xf[1].type = RKNN_TENSOR_FLOAT32;
    zf_xf[1].size = xf[0].size;
    zf_xf[1].fmt = RKNN_TENSOR_NCHW;
    zf_xf[1].buf = xf[0].buf;
    zf_xf[1].pass_through = 0;
    rknn_inputs_set(net_neck_head, 2, zf_xf);

    rknn_run(net_neck_head, nullptr);
    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    for (auto & output : outputs) {
        output.want_float = 1;
        output.is_prealloc = 0;
    }
    rknn_outputs_get(net_neck_head, 2, outputs, nullptr);
    time4.stop();
    time4.show_distance("Update stage ---- output cls_score and bbox_pred extracting cost time");

    time5.start();
    // manually call sigmoid on the output
    std::vector<float> cls_score_sigmoid;

    float* cls_score_data = (float*)outputs[0].buf;
    // print_rknn(cls_score_data, "cls_score_rknn_cpp.txt");
    float* bbox_pred_data = (float*)outputs[1].buf;
    // print_rknn(bbox_pred_data, "bbox_pred_rknn_cpp.txt");
    cls_score_sigmoid.clear();

    int cols = score_size;
    int rows = score_size;

    for (int i = 0; i < cols*rows; i++)   // 18 * 18
    {
        cls_score_sigmoid.push_back(sigmoid(cls_score_data[i]));
    }

    std::vector<float> pred_x1(cols*rows, 0), pred_y1(cols*rows, 0), pred_x2(cols*rows, 0), pred_y2(cols*rows, 0);

    float* bbox_pred_data1 = (float*)outputs[1].buf;
    float* bbox_pred_data2 = (float*)outputs[1].buf + cols*rows;
    float* bbox_pred_data3 = (float*)outputs[1].buf + 2*cols*rows;
    float* bbox_pred_data4 = (float*)outputs[1].buf + 3*cols*rows;
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - bbox_pred_data1[i*cols + j];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - bbox_pred_data2[i*cols + j];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + bbox_pred_data3[i*cols + j];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + bbox_pred_data4[i*cols + j];
        }
    }

    // size penalty (1)
    std::vector<float> w(cols*rows, 0), h(cols*rows, 0);
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * this->penalty_tk);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - this->window_influence) + this->window[i] * this->window_influence;
        if (pscore[i] > maxScore)
        {
            // get max
            maxScore = pscore[i];
            r_max = std::floor(i / rows);
            c_max = ((float)i / rows - r_max) * rows;
        }
    }

    time5.stop();
    time5.show_distance("Update stage ---- postprocess cost time");
    std::cout << "pscore_window max score is: " << pscore[r_max * cols + c_max] << std::endl;

    // to real size
    float pred_x1_real = pred_x1[r_max * cols + c_max]; // pred_x1[r_max, c_max]
    float pred_y1_real = pred_y1[r_max * cols + c_max];
    float pred_x2_real = pred_x2[r_max * cols + c_max];
    float pred_y2_real = pred_y2[r_max * cols + c_max];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs - float(this->instance_size) / 2;
    float diff_ys = pred_ys - float(this->instance_size) / 2;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /=scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr_new = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * this->lr;

    // size rate
    float res_xs = float (target_pos.x) + diff_xs;
    float res_ys = float (target_pos.y) + diff_ys;
    float res_w = pred_w * lr_new + (1 - lr_new) * target_sz.x;
    float res_h = pred_h * lr_new + (1 - lr_new) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr_new) + lr_new * res_w;
    target_sz.y = target_sz.y * (1 - lr_new) + lr_new * res_h;

    rknn_outputs_release(net_neck_head, 2, outputs);
    rknn_outputs_release(net_backbone, 1, xf);
}

void LightTrack::track(const cv::Mat& im)
{
    time_checker time1{};

    float hc_z = target_sz.y + this->context_amount * (target_sz.x + target_sz.y);
    float wc_z = target_sz.x + this->context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  // roi size
    float scale_z = float(this->exemplar_size) / s_z;  // 127/

    float d_search = float(this->instance_size - this->exemplar_size) / 2;  // backbone_model_size - init_model_size = 288-127
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;


    time1.start();
    cv::Mat x_crop;
    x_crop  = get_subwindow_tracking(im, target_pos, this->instance_size, int(s_x));
    time1.stop();
    time1.show_distance("Update stage ---- get subwindow cost time");

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    this->update(x_crop, scale_z);
    target_pos.x = std::max(0, min(im.cols, target_pos.x));
    target_pos.y = std::max(0, min(im.rows, target_pos.y));
    target_sz.x = float(std::max(10, min(im.cols, int(target_sz.x))));
    target_sz.y = float(std::max(10, min(im.rows, int(target_sz.y))));

    std::cout << "track target pos: " << target_pos << std::endl;
    std::cout << "track target_sz: " << target_sz << std::endl;
}

unsigned char *LightTrack::load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    auto *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}

void LightTrack::grids()
{
    /*
    each element of feature map on input search image
    :return: H*W*2 (position for each element)
    */
    int sz = score_size;   // 18

    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = j*total_stride;   // 0~18*16 = 0~288
            this->grid_to_search_y[i*sz+j] = i*total_stride;
        }
    }
}

cv::Mat LightTrack::get_subwindow_tracking(const cv::Mat& im, cv::Point2f pos, int model_sz, int original_sz)
{
    time_checker time1,time2, time3;
    time1.start();
    float c = (float)(original_sz + 1) / 2;
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;

    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;
    time1.stop();
    time1.show_distance("get_subwindow_tracking cost time 1:");

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        time2.start();
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
        //te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)) = im;
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0.f);
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));
        time2.stop();
        time2.show_distance("get_subwindow_tracking cost time 2:");
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    time3.start();
    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));
    time3.stop();
    time3.show_distance("get_subwindow_tracking cost time 3-2:");

    return im_path;
}