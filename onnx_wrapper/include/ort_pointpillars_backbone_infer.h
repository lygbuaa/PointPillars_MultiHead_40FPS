#ifndef __ORT_PILLAR_BACKBONE_INFER_H__
#define __ORT_PILLAR_BACKBONE_INFER_H__

#include "ort_model_infer_base.h"

typedef struct
{
    std::vector<float> pseudo_image;          /** (1,64,512,512) */
}POINTPILLARS_BACKBONE_MODEL_INPUT_t;

typedef struct
{
    std::vector<float> cls_pred_0;       /** (1,32768,1) */
    std::vector<float> cls_pred_12;      /** (1,65536,2) */
    std::vector<float> cls_pred_34;      /** (1,65536,2) */
    std::vector<float> cls_pred_5;       /** (1,32768,1) */
    std::vector<float> cls_pred_67;      /** (1,65536,2) */
    std::vector<float> cls_pred_89;      /** (1,65536,2) */
    std::vector<float> box_preds;        /** (1,327680,9) */
}POINTPILLARS_BACKBONE_MODEL_OUTPUT_t;

class OrtPointPillarsBackboneInfer : public OrtModelInferBase 
{
public:
    OrtPointPillarsBackboneInfer() : OrtModelInferBase("OrtPointPillarsBackboneInfer")
    {
    }

    ~OrtPointPillarsBackboneInfer()
    {
    }

    void RunPointpillarsBackboneModel(POINTPILLARS_BACKBONE_MODEL_INPUT_t& input, POINTPILLARS_BACKBONE_MODEL_OUTPUT_t& output);
    void TestPointpillarsBackboneModel();

private:
    constexpr static size_t MAX_PSEUDO_IMAGE_PIXELS_ = 64*512*512;
};

#endif //__ORT_PILLAR_BACKBONE_INFER_H__