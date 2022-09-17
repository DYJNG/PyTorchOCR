#pragma once

#include "opencv2/core.hpp"  // ""表示从用户自定义的目录下查找 若找不到再从C++安装目录和系统目录下查找
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>   // <>表示从系统文件目录下查找  C++提供的
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <string>
#include <fstream>
#include <numeric>

#include "clipper.h"

using namespace std;

namespace db_postprocess {

class DBPostProcessor {
public:
    void GetContourArea(const std::vector<std::vector<float>> &box, 
                        float unclip_ratio, float &distance);

    cv::RotatedRect UnClip(std::vector<std::vector<float>> box, 
                           const float &unclip_ratio);

    float **Mat2Vec(cv::Mat mat);

    cv::Mat get_affine_transform(const cv::Point2f &center, 
                                 const float img_maxsize, 
                                 const float target_size, 
                                 const int inv);

    cv::Point2f transform_preds(const cv::Mat &warpMat, const cv::Point2f &pt);

    std::vector<std::vector<int>> OrderPointsClockwise(std::vector<std::vector<int>> pts);

    std::vector<std::vector<float>> GetMiniBoxes(cv::RotatedRect box, 
                                                 float &ssid);
    
    float BoxScore(std::vector<cv::Point> contour, cv::Mat pred);

    std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(const cv::Mat pred, 
                                                               const cv::Mat bitmap, 
                                                               const float &box_thresh, 
                                                               const float &det_db_unclip_ratio,
                                                               int src_w, int src_h,
                                                               bool use_padding_resize);

private:
    static bool XsortInt(std::vector<int> a, std::vector<int> b);

    static bool XsortFp32(std::vector<float> a, std::vector<float> b);

    std::vector<std::vector<float>> Mat2Vector(cv::Mat mat);

    inline int _max(int a, int b) { return a >= b ? a : b; }

    inline int _min(int a, int b) { return a >= b ? b : a; }

    template <class T> inline T clamp(T x, T min, T max) {
        if (x > max)
            return max;
        if (x < min)
            return min;
        return x;
    }

    inline float clampf(float x, float min, float max) {
        if (x > max)
            return max;
        if (x < min)
            return min;
        return x;
    }
};

} // namespace db_postprocess
