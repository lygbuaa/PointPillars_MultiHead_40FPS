#include <algorithm>
#include <iostream>
#include <fstream>
#include "nms.h"
#include "postproc.h"

PointpillarsOpsPostProc::PointpillarsOpsPostProc(
    const int num_threads, 
    const float float_min, const float float_max,
    const int num_class, const int num_anchor_per_cls, 
    const std::vector<std::vector<int>> multihead_label_mapping,
    const float score_threshold,  
    const float nms_overlap_threshold, 
    const int nms_pre_maxsize, 
    const int nms_post_maxsize,
    const int num_box_corners,
    const int num_input_box_feature, 
    const int num_output_box_feature
)
:
    num_threads_(num_threads),
    float_min_(float_min),
    float_max_(float_max),
    num_class_(num_class),
    num_anchor_per_cls_(num_anchor_per_cls),
    multihead_label_mapping_(multihead_label_mapping),
    score_threshold_(score_threshold),
    nms_overlap_threshold_(nms_overlap_threshold),
    nms_pre_maxsize_(nms_pre_maxsize),
    nms_post_maxsize_(nms_post_maxsize),
    num_box_corners_(num_box_corners),
    num_input_box_feature_(num_input_box_feature),
    num_output_box_feature_(num_output_box_feature) 
{
    nms_op_ = std::make_shared<PointpillarsOpsNMS>(
        num_threads_,
        num_box_corners_,
        nms_overlap_threshold_
    );
}

PointpillarsOpsPostProc::~PointpillarsOpsPostProc()
{
}

bool PointpillarsOpsPostProc::DoPostProcST(
    std::vector<POINTPILLARS_BBOX3D_t>& bboxes,
    int* const host_filtered_count,
    const float* host_box, 
    const float* host_score
)
{
    const int stride[10] = {0 , 1 , 1 , 5 , 5 , 9 , 10 , 10 , 14 , 14};
    const int offset[10] = {0 , 0 , 1 , 0 , 1 , 0 , 0 , 1 , 0 , 1};
    bboxes.clear();
    for (int class_idx = 0; class_idx < num_class_; ++ class_idx) 
    {   // hardcode for class_map as {0, 12 , 34 , 5 ,67 ,89}
        // init parameter
        host_filtered_count[class_idx] = 0;

        // sigmoid filter
        float host_filtered_score[nms_pre_maxsize_]; // 1000
        float host_filtered_box[nms_pre_maxsize_ * 7]; // 1000 * 7
        for (size_t anchor_idx = 0 ; anchor_idx < num_anchor_per_cls_ ; anchor_idx++)
        {
            float score_upper = 0;
            float score_lower = 0;
            if (class_idx == 0 || class_idx == 5 )
            {
                score_upper =  1 / (1 + exp(-host_score[ stride[class_idx] * num_anchor_per_cls_ + anchor_idx ])); // sigmoid function
            }
            else 
            {
                score_upper =  1 / (1 + exp(-host_score[ stride[class_idx] * num_anchor_per_cls_  + anchor_idx * 2  + offset[class_idx]]));
                score_lower =  1 / (1 + exp(-host_score[ stride[class_idx] * num_anchor_per_cls_  + (num_anchor_per_cls_ + anchor_idx) * 2 + offset[class_idx]]));
            }

            if (score_upper > score_threshold_ && host_filtered_count[class_idx] < nms_pre_maxsize_)  // filter out boxes which threshold less than score_threshold
            {
                host_filtered_score[host_filtered_count[class_idx]] = score_upper;
                for (size_t dim_idx = 0 ; dim_idx < 7 ; dim_idx++) // dim_idx = {x,y,z,dx,dy,dz,yaw}
                { 
                    host_filtered_box[host_filtered_count[class_idx] * 7 + dim_idx] \
                    =  host_box[ class_idx * num_anchor_per_cls_ * num_output_box_feature_ + anchor_idx * num_output_box_feature_ + dim_idx];
                }
                host_filtered_count[class_idx] += 1;
            }

            if (score_lower > score_threshold_ && host_filtered_count[class_idx] < nms_pre_maxsize_)  // filter out boxes which threshold less than score_threshold
            {
                host_filtered_score[host_filtered_count[class_idx]] = score_lower;
                for (size_t dim_idx = 0 ; dim_idx < 7 ; dim_idx++) // dim_idx = {x,y,z,dx,dy,dz}
                { 
                    host_filtered_box[host_filtered_count[class_idx] * 7 + dim_idx] \
                    =  host_box[ class_idx * num_anchor_per_cls_ * num_output_box_feature_ + anchor_idx * num_output_box_feature_+ dim_idx];
                }
                host_filtered_count[class_idx] += 1;
            }

        }
        if(host_filtered_count[class_idx] <= 0) continue;

        // sort boxes (topk)
        float host_sorted_filtered_box[host_filtered_count[class_idx] * 7];
        float host_sorted_filtered_score[host_filtered_count[class_idx]];
        int host_sorted_filtered_indexes[host_filtered_count[class_idx]];
        for (int i = 0 ; i < host_filtered_count[class_idx] ; i++)
        {
            host_sorted_filtered_indexes[i] = i;
        }

        QuicksortKernel(host_filtered_score, host_sorted_filtered_indexes, host_filtered_count[class_idx]);
        for (int ith_box = 0 ; ith_box  < host_filtered_count[class_idx] ; ++ith_box) 
        {
            host_sorted_filtered_score[ith_box] = host_filtered_score[ith_box];
            host_sorted_filtered_box[ith_box * 7 + 0] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 0];
            host_sorted_filtered_box[ith_box * 7 + 1] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 1];
            host_sorted_filtered_box[ith_box * 7 + 2] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 2];
            host_sorted_filtered_box[ith_box * 7 + 3] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 3];
            host_sorted_filtered_box[ith_box * 7 + 4] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 4];
            host_sorted_filtered_box[ith_box * 7 + 5] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 5];
            host_sorted_filtered_box[ith_box * 7 + 6] = host_filtered_box[host_sorted_filtered_indexes[ith_box] * 7 + 6];
        }

        // host to device for nms cuda
        // In fact, this cuda calc is also not necessary. 
        // After each category is filtered by sigmoid, there are only about 100, up to 1000 boxes left behind. 
        // Use CUDA_NMS will never faster than CPU_NMS
        // TODO : use cpu_nms replace cuda_nms
        float* dev_sorted_filtered_box;
        float* dev_sorted_filtered_score;
        int det_num_boxes_per_class = 0;

        int num_box_for_nms = std::min(nms_pre_maxsize_, host_filtered_count[class_idx]);
        long keep_inds[num_box_for_nms]; // index of kept box
        memset(keep_inds, 0, num_box_for_nms * sizeof(int));

        bool ret = nms_op_ -> DoNmsST(
            &det_num_boxes_per_class,
            keep_inds,
            num_box_for_nms,
            host_sorted_filtered_box
        );
        if(!ret)
        {
            return ret;
        }
        RLOGI("DoNmsST for cls[%d] out_num_to_keep: %d", class_idx, det_num_boxes_per_class);

        // int det_num_filtered_boxes_pre_class = 0;
        for (int box_idx = 0; box_idx < det_num_boxes_per_class; ++box_idx)
        {
            POINTPILLARS_BBOX3D_t box;
            box.x = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 0];
            box.y = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 1];
            box.z = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 2];
            box.dx = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 3];
            box.dy = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 4];
            box.dz = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 5];
            box.yaw = host_sorted_filtered_box[keep_inds[box_idx] * 7 + 6];
            box.score = host_sorted_filtered_score[keep_inds[box_idx]];
            box.cls = class_idx;
            bboxes.emplace_back(box);
        }
    }
    return true;
}

void PointpillarsOpsPostProc::QuicksortWarp(float* score, int* index, const int start, const int end)
{
    if (start>=end) return;
    float pivot=score[end];
    float value_swp;
    int index_swp;
    //set a pointer to divide array into two parts
    //one part is smaller than pivot and another larger
    int pointer=start;
    for (int i = start; i < end; i++) 
    {
        if (score[i] > pivot) 
        {
            if (pointer!=i) 
            {
                //swap score[i] with score[pointer]
                //score[pointer] behind larger than pivot
                SwapWarp<float>(score[i], score[pointer], value_swp);
                SwapWarp<int>(index[i], index[pointer], index_swp);
            }
            pointer++;
        }
    }
    //swap back pivot to proper position
    SwapWarp<float>(score[end], score[pointer], value_swp);
    SwapWarp<int>(index[end], index[pointer], index_swp);
    QuicksortWarp(score, index, start, pointer-1);
    QuicksortWarp(score, index, pointer+1, end);
    return ;
}

void PointpillarsOpsPostProc::QuicksortKernel(float* score, int* indexes, const int len)
{
    QuicksortWarp(score,indexes ,0,len-1);
}

void PointpillarsOpsPostProc::SaveBox3dToFile(
    const std::string& file_path, 
    const std::vector<POINTPILLARS_BBOX3D_t>& bboxes
)
{
    std::ofstream ofFile;
    ofFile.open(file_path.c_str() , std::ios::out);  
    if(ofFile.is_open())
    {
        for (int i=0; i < bboxes.size(); ++i) 
        {
            const POINTPILLARS_BBOX3D_t& box_i = bboxes[i];
            ofFile << box_i.x << " " << box_i.y << " " << box_i.z << " " << box_i.dx << " " << box_i.dy << " " << box_i.dz << " " << box_i.yaw << std::endl;
        }
    }
    ofFile.close();
    return;
};