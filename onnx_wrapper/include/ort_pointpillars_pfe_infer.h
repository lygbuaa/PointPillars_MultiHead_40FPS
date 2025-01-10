#ifndef __ORT_PILLAR_PFE_INFER_H__
#define __ORT_PILLAR_PFE_INFER_H__

#include "ort_model_infer_base.h"

typedef struct
{
    std::vector<float> pillar_features;         /** (30000,20,11) */
}POINTPILLARS_PFE_MODEL_INPUT_t;

typedef struct
{
    std::vector<float> learned_features;       /** (30000,64) */
}POINTPILLARS_PFE_MODEL_OUTPUT_t;

class OrtPointPillarsPfeInfer : public OrtModelInferBase
{
public:
    OrtPointPillarsPfeInfer() : OrtModelInferBase("OrtPointPillarsPfeInfer")
    {
    }

    ~OrtPointPillarsPfeInfer()
    {
    }

    void RunPointpillarsPfeModel(POINTPILLARS_PFE_MODEL_INPUT_t& input, POINTPILLARS_PFE_MODEL_OUTPUT_t& output);
    void TestPointpillarsPfeModel();

private:
    constexpr static size_t MAX_NUM_PILLARS_ = 30000*20*11;
};

#endif //__ORT_PILLAR_PFE_INFER_H__