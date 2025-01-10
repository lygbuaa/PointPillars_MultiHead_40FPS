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