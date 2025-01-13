#include "ort_pointpillars_backbone_infer.h"


/* img should be rgb format */
void OrtPointPillarsBackboneInfer::RunPointpillarsBackboneModel(POINTPILLARS_BACKBONE_MODEL_INPUT_t& input, POINTPILLARS_BACKBONE_MODEL_OUTPUT_t& output)
{
    HANG_STOPWATCH();
    /* prepare input data */
    std::vector<const char*>& input_node_names = g_model_s_.input_node_names;
    std::vector<std::vector<int64_t>>& input_node_dims = g_model_s_.input_node_dims;
    std::vector<ONNXTensorElementDataType>& input_types = g_model_s_.input_types;
    std::vector<OrtValue*>& input_tensors = g_model_s_.input_tensors;

    std::vector<const char*>& output_node_names = g_model_s_.output_node_names;
    std::vector<std::vector<int64_t>>& output_node_dims = g_model_s_.output_node_dims;
    std::vector<ONNXTensorElementDataType>& output_types = g_model_s_.output_types;
    std::vector<OrtValue*>& output_tensors = g_model_s_.output_tensors;

    /* move input vector into input_tensors */
    OrtMemoryInfo* memory_info;
    CheckStatus(g_ort_->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));
    CheckStatus(g_ort_->CreateTensorWithDataAsOrtValue(
                    memory_info, reinterpret_cast<void*>(input.pseudo_image.data()), sizeof(float)*input.pseudo_image.size(),
                    input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
    g_ort_->ReleaseMemoryInfo(memory_info);
    RLOGI("CreateTensorWithDataAsOrtValue for pseudo_image: %ld", sizeof(float)*input.pseudo_image.size());

    /* do inference */
    CheckStatus(g_ort_->Run(g_model_s_.sess, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                    input_tensors.size(), output_node_names.data(), output_node_names.size(), output_tensors.data()));

    /* postprocess */
    // assert (output_node_names.size() == 1);
    // RLOGI("retrieve output[0]: %s\n", output_node_names[0]);
    float* cls_pred_0;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[0], reinterpret_cast<void**>(&cls_pred_0)));
    size_t cls_pred_0_size = 1;
    for(size_t k=0; k<output_node_dims[0].size(); k++)
    {
        cls_pred_0_size *= output_node_dims[0][k];
    }
    output.cls_pred_0.assign(cls_pred_0, cls_pred_0+cls_pred_0_size);
    RLOGI("cls_pred_0_size: %ld", cls_pred_0_size);

    float* cls_pred_12;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&cls_pred_12)));
    size_t cls_pred_12_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        cls_pred_12_size *= output_node_dims[2][k];
    }
    output.cls_pred_12.assign(cls_pred_12, cls_pred_12+cls_pred_12_size);
    RLOGI("cls_pred_12: %ld", cls_pred_12_size);

    float* cls_pred_34;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&cls_pred_34)));
    size_t cls_pred_34_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        cls_pred_34_size *= output_node_dims[2][k];
    }
    output.cls_pred_34.assign(cls_pred_34, cls_pred_34+cls_pred_34_size);
    RLOGI("cls_pred_34: %ld", cls_pred_34_size);

    float* cls_pred_5;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&cls_pred_5)));
    size_t cls_pred_5_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        cls_pred_5_size *= output_node_dims[2][k];
    }
    output.cls_pred_5.assign(cls_pred_5, cls_pred_5+cls_pred_5_size);
    RLOGI("cls_pred_5: %ld", cls_pred_5_size);

    float* cls_pred_67;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&cls_pred_67)));
    size_t cls_pred_67_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        cls_pred_67_size *= output_node_dims[2][k];
    }
    output.cls_pred_67.assign(cls_pred_67, cls_pred_67+cls_pred_67_size);
    RLOGI("cls_pred_67: %ld", cls_pred_67_size);

    float* cls_pred_89;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&cls_pred_89)));
    size_t cls_pred_89_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        cls_pred_89_size *= output_node_dims[2][k];
    }
    output.cls_pred_89.assign(cls_pred_89, cls_pred_89+cls_pred_89_size);
    RLOGI("cls_pred_89: %ld", cls_pred_89_size);

    float* box_preds;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[2], reinterpret_cast<void**>(&box_preds)));
    size_t box_preds_size = 1;
    for(size_t k=0; k<output_node_dims[2].size(); k++)
    {
        box_preds_size *= output_node_dims[2][k];
    }
    output.box_preds.assign(box_preds, box_preds+box_preds_size);
    RLOGI("box_preds: %ld", box_preds_size);
}

void OrtPointPillarsBackboneInfer::TestPointpillarsBackboneModel()
{
    RLOGI("TestPointpillarsBackboneModel\n");
    POINTPILLARS_BACKBONE_MODEL_INPUT_t input;
    POINTPILLARS_BACKBONE_MODEL_OUTPUT_t output;

    for(int i=1; i<10; i++)
    {
        input.pseudo_image.resize(MAX_PSEUDO_IMAGE_PIXELS_);
        RunPointpillarsBackboneModel(input, output);
    }
}

