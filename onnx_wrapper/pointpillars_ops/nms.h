#ifndef __POINTPILLARS_OPS_NMS_H__
#define __POINTPILLARS_OPS_NMS_H__

#include <cstring>
#include <cmath>
#include <vector>
#include "logging_utils.h"

class PointpillarsOpsNMS
{
public:
    PointpillarsOpsNMS(
        const int num_threads, 
        const int num_box_corners,
        const float nms_overlap_threshold
    );
    ~PointpillarsOpsNMS();

    bool DoNmsST(
        int* const out_num_to_keep,
        long* const out_keep_inds, 
        const int in_sorted_box_count, 
        const float* in_sorted_box_for_nms
    );

private:
    const int num_threads_;
    const int num_box_corners_;
    const float nms_overlap_threshold_;
};

#endif