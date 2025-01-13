#include <iostream>
#include <fstream>
#include "pointpillars_pipeline.h"


bool PointPillarsPipeline::Init(const char* pfe_model_path, const char* backbone_model_path)
{
    ort_pfe_model_ = std::make_shared<OrtPointPillarsPfeInfer>();
    ort_pfe_model_->LoadONNXModel(pfe_model_path);

    ort_backbone_model_ = std::make_shared<OrtPointPillarsBackboneInfer>();
    ort_backbone_model_->LoadONNXModel(backbone_model_path);
    return true;
}

bool PointPillarsPipeline::Stop()
{
    return true;
}

void PointPillarsPipeline::RunTest()
{
    ort_pfe_model_->TestPointpillarsPfeModel();
    ort_backbone_model_->TestPointpillarsBackboneModel();
}

size_t PointPillarsPipeline::LoadPCDFile(const char* pcd_txt_path, const int num_feature)
{
    if(strlen(pcd_txt_path) < 3)
    {
        LOGPF("pcd_txt_path invalid.");
        return 0;
    }
    std::ifstream fs_pcd;
    fs_pcd.open(pcd_txt_path);
    assert(fs_pcd.is_open());
    std::vector<float> temp_points;

    std::string line;
    size_t points_counter = 0;
    while(std::getline(fs_pcd, line))
    {
        points_counter ++;
        // LOGPF("line[%d]: %s", points_counter, line.c_str());
        float x, y, z, i, r;
        sscanf(line.c_str(), "%e %e %e %e %e\n", &x, &y, &z, &i, &r);
        // LOGPF("point[%d] x=%f, y=%f, z=%f, i=%f, r=%f", x, y, z, i, r);
        temp_points.push_back(x);
        temp_points.push_back(y);
        temp_points.push_back(z);
        temp_points.push_back(i);
        temp_points.push_back(r);
    }

    fs_pcd.close();
    LOGPF("temp_points size: %ld", temp_points.size());
    size_t points_array_size = num_feature * points_counter;

    pcd_array_ = std::shared_ptr<float>(new float[points_array_size]);
    float* const points_array = pcd_array_.get();
    for(int i=0; i<points_counter; ++i)
    {
        for(int j=0; j<num_feature; j++)
        {
            size_t out_idx = i*num_feature + j;
            size_t tmp_idx = i*5 + j;
            points_array[out_idx] = temp_points[tmp_idx];
        }
    }

    return points_counter;
}