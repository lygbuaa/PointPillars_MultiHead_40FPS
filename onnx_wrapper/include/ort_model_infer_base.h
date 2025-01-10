#ifndef __ORT_MODEL_INFER_BASE_H__
#define __ORT_MODEL_INFER_BASE_H__

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include "logging_utils.h"


class OrtModelInferBase
{
public:
/* replaced by CheckStatus() */
    #define ORT_ABORT_ON_ERROR(expr)                             \
    do {                                                       \
        OrtStatus* onnx_status = (expr);                         \
        if (onnx_status != NULL) {                               \
            const char* msg = g_ort_->GetErrorMessage(onnx_status); \
            fprintf(stderr, "%s\n", msg);                          \
            g_ort_->ReleaseStatus(onnx_status);                    \
            abort();                                               \
        }                                                        \
    } while (0);

    typedef struct
    {
        OrtSession* sess = nullptr;
        std::vector<const char*> input_node_names;
        std::vector<std::vector<int64_t>> input_node_dims;
        std::vector<ONNXTensorElementDataType> input_types;
        std::vector<OrtValue*> input_tensors;
        std::vector<const char*> output_node_names;
        std::vector<std::vector<int64_t>> output_node_dims;
        std::vector<ONNXTensorElementDataType> output_types;
        std::vector<OrtValue*> output_tensors;
    }ORT_S_t;

public:
    OrtModelInferBase(const char* logid = "OrtModelInferBase")
    {
        InitOrt(logid);
    }

    ~OrtModelInferBase()
    {
        DestroyOrt();
    }

    bool LoadONNXModel(const std::string& model_path);

protected:
    void LoadModel(const std::string& model_path, ORT_S_t& model_s);
    bool CheckStatus(OrtStatus* status);
    bool InitOrt(const char* logid = "OrtModelInferBase");
    void DestroyOrt();
    void VerifyInputOutputCount(OrtSession* sess);
    int EnableCuda(OrtSessionOptions* session_options);

protected:
    const OrtApi* g_ort_ = nullptr;
    const OrtApiBase* g_ort_base_ = nullptr;
    OrtEnv* env_ = nullptr;
    OrtSessionOptions* session_options_ = nullptr;

    ORT_S_t g_model_s_;
};

#endif //__ORT_MODEL_INFER_BASE_H__