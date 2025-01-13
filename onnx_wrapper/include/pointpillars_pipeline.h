#ifndef __POINTPILLARS_PIPELINE_H__
#define __POINTPILLARS_PIPELINE_H__

#include "ort_pointpillars_pfe_infer.h"
#include "ort_pointpillars_backbone_infer.h"

class PointPillarsPipeline
{
public:
    PointPillarsPipeline()
    {
    }

    ~PointPillarsPipeline()
    {
    }

    size_t LoadPCDFile(const char* pcd_txt_path, const int num_feature=4);
    bool Init(const char* pfe_model_path, const char* backbone_model_path);
    bool Stop();

    void RunTest();

private:
    std::shared_ptr<OrtPointPillarsPfeInfer> ort_pfe_model_;
    std::shared_ptr<OrtPointPillarsBackboneInfer> ort_backbone_model_;
    std::shared_ptr<float> pcd_array_;

};

#endif //__POINTPILLARS_PIPELINE_H__