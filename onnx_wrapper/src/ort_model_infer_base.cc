#include "ort_model_infer_base.h"


bool OrtModelInferBase::LoadONNXModel(const std::string& model_path)
{
    try{
        LoadModel(model_path, g_model_s_);
    }catch(Ort::Exception& e){
        LOGPF("ort exception: %s\n", e.what());
        return false;
    }
    return true;
}

void OrtModelInferBase::LoadModel(const std::string& model_path, ORT_S_t& model_s)
{
    LOGPF("loading onnx model: %s\n", model_path.c_str());

    CheckStatus(g_ort_->CreateSession(env_, model_path.c_str(), session_options_, &model_s.sess));
    LOGPF("CreateSession sucess.");

    OrtAllocator* allocator;
    CheckStatus(g_ort_->GetAllocatorWithDefaultOptions(&allocator));
    LOGPF("GetAllocatorWithDefaultOptions sucess.");

    size_t num_input_nodes;
    CheckStatus(g_ort_->SessionGetInputCount(model_s.sess, &num_input_nodes));
    LOGPF("SessionGetInputCount sucess: %ld", num_input_nodes);

    std::vector<const char*>& input_node_names = model_s.input_node_names;
    std::vector<std::vector<int64_t>>& input_node_dims = model_s.input_node_dims;
    std::vector<ONNXTensorElementDataType>& input_types = model_s.input_types;
    std::vector<OrtValue*>& input_tensors = model_s.input_tensors;

    input_node_names.resize(num_input_nodes);
    input_node_dims.resize(num_input_nodes);
    input_types.resize(num_input_nodes);
    input_tensors.resize(num_input_nodes);

    for (size_t i = 0; i < num_input_nodes; i++) 
    {
        // Get input node names
        char* input_name;
        CheckStatus(g_ort_->SessionGetInputName(model_s.sess, i, allocator, &input_name));
        input_node_names[i] = input_name;
        LOGPF("input_node_names[%d]:  %s", i, input_name);

        // Get input node types
        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort_->SessionGetInputTypeInfo(model_s.sess, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort_->GetTensorElementType(tensor_info, &type));
        input_types[i] = type;
        LOGPF("input_types[%d]: %d", i, (int)(type));

        // Get input shapes/dims
        size_t num_dims;
        CheckStatus(g_ort_->GetDimensionsCount(tensor_info, &num_dims));
        LOGPF("GetDimensionsCount[%d]: %d", i, (int)(num_dims));

        input_node_dims[i].resize(num_dims);
        CheckStatus(g_ort_->GetDimensions(tensor_info, input_node_dims[i].data(), num_dims));
        for(size_t j=0; j<num_dims; j++)
        {
            LOGPF("GetDimensions[%d][%d]: %ld", i, j, input_node_dims[i][j]);
        }

        // size_t tensor_size;
        // CheckStatus(g_ort_->GetTensorShapeElementCount(tensor_info, &tensor_size));
        // LOGPF("GetTensorShapeElementCount[%d]: %ld", i, tensor_size);

        std::string dimstr="(";
        for(int k=0; k<num_dims; ++k){
            dimstr += std::to_string(input_node_dims[i][k]);
            dimstr += ",";
        }
        dimstr += ")";

        /* print input tensor information */
        LOGPF("input[%ld]-%s, type: %d, dims: %s\n", i, input_name, type, dimstr.c_str());

        if (typeinfo) g_ort_->ReleaseTypeInfo(typeinfo);
    }

    size_t num_output_nodes;
    std::vector<const char*>& output_node_names = model_s.output_node_names;
    std::vector<std::vector<int64_t>>& output_node_dims = model_s.output_node_dims;
    std::vector<ONNXTensorElementDataType>& output_types = model_s.output_types;
    std::vector<OrtValue*>& output_tensors = model_s.output_tensors;

    CheckStatus(g_ort_->SessionGetOutputCount(model_s.sess, &num_output_nodes));
    LOGPF("num_output_nodes: %ld\n", num_output_nodes);
    output_node_names.resize(num_output_nodes);
    output_node_dims.resize(num_output_nodes);
    output_tensors.resize(num_output_nodes);
    output_types.resize(num_output_nodes);

    for (size_t i = 0; i < num_output_nodes; i++) 
    {
        // Get output node names
        char* output_name;
        CheckStatus(g_ort_->SessionGetOutputName(model_s.sess, i, allocator, &output_name));
        output_node_names[i] = output_name;
        // LOGPF("%ld-output_name: %s\n", i, output_name);

        OrtTypeInfo* typeinfo;
        CheckStatus(g_ort_->SessionGetOutputTypeInfo(model_s.sess, i, &typeinfo));
        const OrtTensorTypeAndShapeInfo* tensor_info;
        CheckStatus(g_ort_->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
        ONNXTensorElementDataType type;
        CheckStatus(g_ort_->GetTensorElementType(tensor_info, &type));
        output_types[i] = type;
        // LOGPF("%ld-type: %d\n", i, type);

        // Get output shapes/dims
        size_t num_dims;
        CheckStatus(g_ort_->GetDimensionsCount(tensor_info, &num_dims));
        output_node_dims[i].resize(num_dims);
        CheckStatus(g_ort_->GetDimensions(tensor_info, (int64_t*)output_node_dims[i].data(), num_dims));
        // LOGPF("%ld-num_dims: %ld\n", i, num_dims);

        /* when it's variable output, tensor_size could be negative, so tensor_size will overflow */
        // size_t tensor_size;
        // CheckStatus(g_ort_->GetTensorShapeElementCount(tensor_info, &tensor_size));
        // LOGPF("%ld-tensor_size: %ld\n", i, tensor_size);

        std::string dimstr="(";
        for(int k=0; k<num_dims; ++k){
            dimstr += std::to_string(output_node_dims[i][k]);
            dimstr += ",";
        }
        dimstr += ")";
        /* print output tensor information */
        LOGPF("output[%ld]-%s, type: %d, dims: %s\n", i, output_name, type, dimstr.c_str());

        if (typeinfo) g_ort_->ReleaseTypeInfo(typeinfo);
    }
}

bool OrtModelInferBase::CheckStatus(OrtStatus* status) 
{
    if (status != nullptr) {
        const char* msg = g_ort_->GetErrorMessage(status);
        std::cerr << msg << std::endl;
        // g_ort_->ReleaseStatus(status);
        throw Ort::Exception(msg, OrtErrorCode::ORT_EP_FAIL);
    }
    return true;
}

bool OrtModelInferBase::InitOrt(const char* logid)
{
    g_ort_base_ = OrtGetApiBase();
    if (!g_ort_base_){
        LOGPF("Failed to OrtGetApiBase.\n");
        return false;
    }

    LOGPF("ort version: %s\n", g_ort_base_->GetVersionString());

    g_ort_ = g_ort_base_->GetApi(ORT_API_VERSION);
    if (!g_ort_) {
        LOGPF("Failed to init ONNX Runtime engine.\n");
        return false;
    }

    CheckStatus(g_ort_->CreateEnv(ORT_LOGGING_LEVEL_INFO, logid, &env_));
    if (!env_) {
        LOGPF("Failed to CreateEnv.\n");
        return false;
    }

    /* use default session option is ok */
    CheckStatus(g_ort_->CreateSessionOptions(&session_options_));
    // if(EnableCuda(session_options_)<0){
    //     return false;
    // }
    // CheckStatus(g_ort_->SetIntraOpNumThreads(session_options_, 1));
    // CheckStatus(g_ort_->SetSessionGraphOptimizationLevel(session_options_, ORT_ENABLE_ALL));
    // std::vector<const char*> options_keys = {"runtime", "buffer_type"};
    // std::vector<const char*> options_values = {backend.c_str(), "FLOAT"};  // set to TF8 if use quantized data
    // CheckStatus(g_ort_->SessionOptionsAppendExecutionProvider(session_options_, "SNPE", options_keys.data(), options_values.data(), options_keys.size()));

    return true;
}

void OrtModelInferBase::DestroyOrt()
{
    if(session_options_) g_ort_->ReleaseSessionOptions(session_options_);
    if(g_model_s_.sess) g_ort_->ReleaseSession(g_model_s_.sess);
    if(env_) g_ort_->ReleaseEnv(env_);
}

void OrtModelInferBase::VerifyInputOutputCount(OrtSession* sess) 
{
    size_t count;
    CheckStatus(g_ort_->SessionGetInputCount(sess, &count));
    if(count != 1)
    {
        LOGPF("SessionGetInputCount: %ld", count);
        abort();
    }
    CheckStatus(g_ort_->SessionGetOutputCount(sess, &count));
    if(count != 1)
    {
        LOGPF("SessionGetInputCount: %ld", count);
        abort();
    }
}

int OrtModelInferBase::EnableCuda(OrtSessionOptions* session_options) 
{
    // OrtCUDAProviderOptions is a C struct. C programming language doesn't have constructors/destructors.
    OrtCUDAProviderOptions o;
    // Here we use memset to initialize every field of the above data struct to zero.
    memset(&o, 0, sizeof(o));
    // But is zero a valid value for every variable? Not quite. It is not guaranteed. In the other words: does every enum
    // type contain zero? The following line can be omitted because EXHAUSTIVE is mapped to zero in onnxruntime_c_api.h.
    o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    o.gpu_mem_limit = SIZE_MAX;
    OrtStatus* onnx_status = g_ort_->SessionOptionsAppendExecutionProvider_CUDA(session_options, &o);
    if (onnx_status != NULL) {
        const char* msg = g_ort_->GetErrorMessage(onnx_status);
        LOGPF("%s\n", msg);
        g_ort_->ReleaseStatus(onnx_status);
        return -1;
    }
    return 0;
}
