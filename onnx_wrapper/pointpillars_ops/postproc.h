#ifndef __POINTPILLARS_OPS_POSTPROC_H__
#define __POINTPILLARS_OPS_POSTPROC_H__

#include <cstring>
#include <cmath>
#include <vector>
#include "logging_utils.h"
#include "nms.h"

// """
// bbox: [x, y, z, dx, dy, dz, yaw]
// """
// # 8 corners: np.array = n*8*3(x, y, z)
// #         7 -------- 6
// #        /|         /|
// #       4 -------- 5 .
// #       | |        | |
// #       . 3 -------- 2
// #       |/         |/
// #       0 -------- 1

// #             ^ dx(l)
// #             |
// #             |
// #             |
// # dy(w)       |
// # <-----------O

typedef struct
{
    float x;    //center.x
    float y;    //center.y
    float z;    //center.z
    float dx;   //width
    float dy;   //length
    float dz;   //height
    float yaw;  //yaw in rad
    int cls;    //class type
    float score;
}POINTPILLARS_BBOX3D_t;

class PointpillarsOpsPostProc
{
public:
    PointpillarsOpsPostProc(
        const int num_threads, 
        const float float_min, 
        const float float_max,
        const int num_class, 
        const int num_anchor_per_cls, 
        const std::vector<std::vector<int>> multihead_label_mapping,
        const float score_threshold,  
        const float nms_overlap_threshold, 
        const int nms_pre_maxsize, 
        const int nms_post_maxsize,
        const int num_box_corners,
        const int num_input_box_feature, 
        const int num_output_box_feature
    );
    ~PointpillarsOpsPostProc();

    bool DoPostProcST(
        std::vector<POINTPILLARS_BBOX3D_t>& bboxes,
        int* const host_filtered_count,
        const float* host_box, 
        const float* host_score
    );

    static void SaveBox3dToFile(
        const std::string& file_path,
        const std::vector<POINTPILLARS_BBOX3D_t>& bboxes
    );

private:
    template <typename T>
    void SwapWarp(T& a , T& b , T& swp)
    {
        swp=a;
        a=b;
        b=swp;
    }

    void QuicksortWarp(float* score, int* index, const int start, const int end);
    void QuicksortKernel(float* score, int* indexes, const int len);

private:
    const int num_threads_;
    const float float_min_;
    const float float_max_;
    const int num_class_;
    const int num_anchor_per_cls_;
    const float score_threshold_;
    const float nms_overlap_threshold_;
    const int nms_pre_maxsize_;
    const int nms_post_maxsize_;
    const int num_box_corners_;
    const int num_input_box_feature_;
    const int num_output_box_feature_;
    const std::vector<std::vector<int>> multihead_label_mapping_;

    std::shared_ptr<PointpillarsOpsNMS> nms_op_;
};

#endif