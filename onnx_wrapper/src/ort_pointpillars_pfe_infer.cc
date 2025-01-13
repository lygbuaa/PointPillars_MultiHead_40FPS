#include "ort_pointpillars_pfe_infer.h"


/* img should be rgb format */
void OrtPointPillarsPfeInfer::RunPointpillarsPfeModel(POINTPILLARS_PFE_MODEL_INPUT_t& input, POINTPILLARS_PFE_MODEL_OUTPUT_t& output)
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
                    memory_info, reinterpret_cast<void*>(input.pillar_features.data()), sizeof(float)*input.pillar_features.size(),
                    input_node_dims[0].data(), input_node_dims[0].size(), input_types[0], &input_tensors[0]));
    g_ort_->ReleaseMemoryInfo(memory_info);
    RLOGI("CreateTensorWithDataAsOrtValue for pillar_features: %ld", sizeof(float)*input.pillar_features.size());

    /* do inference */
    CheckStatus(g_ort_->Run(g_model_s_.sess, nullptr, input_node_names.data(), (const OrtValue* const*)input_tensors.data(),
                    input_tensors.size(), output_node_names.data(), output_node_names.size(), output_tensors.data()));

    /* postprocess */
    // assert (output_node_names.size() == 1);
    // RLOGI("retrieve output[0]: %s\n", output_node_names[0]);
    float* learned_features;
    CheckStatus(g_ort_->GetTensorMutableData(output_tensors[0], reinterpret_cast<void**>(&learned_features)));
    size_t learned_features_size = 1;
    for(size_t k=0; k<output_node_dims[0].size(); k++)
    {
        learned_features_size *= output_node_dims[0][k];
    }
    output.learned_features.assign(learned_features, learned_features+learned_features_size);
    RLOGI("learned_features_size: %ld", learned_features_size);
}

void OrtPointPillarsPfeInfer::TestPointpillarsPfeModel()
{
    RLOGI("TestPointpillarsPfeModel\n");
    POINTPILLARS_PFE_MODEL_INPUT_t input;
    POINTPILLARS_PFE_MODEL_OUTPUT_t output;

    for(int i=1; i<10; i++)
    {
        input.pillar_features.resize(MAX_NUM_PILLARS_);
        RunPointpillarsPfeModel(input, output);
    }
}

