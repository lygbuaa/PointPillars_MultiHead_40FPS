#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "pointpillars_pipeline.h"

bool PointPillarsPipeline::Init(const char* global_config_path, const char* model_config_path)
{
    LoadGlobalConfigs(global_config_path);
    LoadModelConfigs(model_config_path);

    ort_pfe_model_ = std::make_shared<OrtPointPillarsPfeInfer>();
    if(!ort_pfe_model_->LoadONNXModel(global_params_.PfeOnnxFile))
    {
        RLOGE("load onnx model %s failed.", global_params_.PfeOnnxFile.c_str());
        return false;
    }

    ort_backbone_model_ = std::make_shared<OrtPointPillarsBackboneInfer>();
    if(!ort_backbone_model_->LoadONNXModel(global_params_.BackboneOnnxFile))
    {
        RLOGE("load onnx model %s failed.", global_params_.BackboneOnnxFile.c_str());
        return false;
    }

    AllocBuffers();
    ClearBuffers();

    preproc_op_ = std::make_shared<PointpillarsOpsPreProc>(
        model_params_.kNumThreads,
        model_params_.kMaxNumPillars,
        model_params_.kMaxNumPointsPerPillar,
        model_params_.kNumPointFeature,
        model_params_.kNumIndsForScan,
        model_params_.kGridXSize,
        model_params_.kGridYSize, 
        model_params_.kGridZSize,
        model_params_.kPillarXSize,
        model_params_.kPillarYSize, 
        model_params_.kPillarZSize,
        model_params_.kMinXRange, 
        model_params_.kMinYRange, 
        model_params_.kMinZRange
    );
    scatter_op_ = std::make_shared<PointpillarsOpsScatter>();
    postproc_op_ = std::make_shared<PointpillarsOpsPostProc>();
    nms_op_ = std::make_shared<PointpillarsOpsNMS>();

    return true;
}

bool PointPillarsPipeline::Stop()
{
    FreeBuffers();
    return true;
}

void PointPillarsPipeline::RunTest()
{
    ort_pfe_model_->TestPointpillarsPfeModel();
    ort_backbone_model_->TestPointpillarsBackboneModel();
}

bool PointPillarsPipeline::RunPipeline()
{
    return true;
}

size_t PointPillarsPipeline::LoadPCDFile(const char* pcd_txt_path, const int num_feature)
{
    if(strlen(pcd_txt_path) < 3)
    {
        RLOGE("pcd_txt_path invalid.");
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
        // RLOGI("line[%d]: %s", points_counter, line.c_str());
        float x, y, z, i, r;
        sscanf(line.c_str(), "%e %e %e %e %e\n", &x, &y, &z, &i, &r);
        // RLOGI("point[%d] x=%f, y=%f, z=%f, i=%f, r=%f", x, y, z, i, r);
        temp_points.push_back(x);
        temp_points.push_back(y);
        temp_points.push_back(z);
        temp_points.push_back(i);
        // temp_points.push_back(r);
        /** r stands for scan-round in pointpillars, 
         *  in 10-sweeps file, it varies from 0.0, 0.05, 0.10, ... 0.45,
         *  so in single sweep file, let r=0 works fine. */
        temp_points.push_back(0.0f); 
    }

    fs_pcd.close();
    RLOGI("LoadPCDFile temp_points size: %ld", temp_points.size());
    size_t points_array_size = num_feature * points_counter;

    pcd_array_ = std::shared_ptr<float>(new float[points_array_size]);
    float* const points_array = pcd_array_.get();
    for(size_t i=0; i<points_counter; ++i)
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

void PointPillarsPipeline::LoadGlobalConfigs(const char* global_yaml_file)
{
    YAML::Node config = YAML::LoadFile(global_yaml_file);
    global_params_.BoxFeature = config["BoxFeature"].as<int>();
    global_params_.ScoreThreshold = config["ScoreThreshold"].as<float>();
    global_params_.NmsOverlapThreshold = config["NmsOverlapThreshold"].as<float>();
    global_params_.PfeOnnxFile = config["PfeOnnx"].as<std::string>();
    global_params_.BackboneOnnxFile = config["BackboneOnnx"].as<std::string>();
    global_params_.InputPCDFile = config["InputFile"].as<std::string>();
    global_params_.OutputDetsFile = config["OutputFile"].as<std::string>();
    RLOGI("BoxFeature: %d, ScoreThreshold: %.2f, NmsOverlapThreshold: %.2f, PfeOnnxFile: %s, BackboneOnnxFile: %s, InputPCDFile: %s, OutputDetsFile: %s",\ 
        global_params_.BoxFeature, global_params_.ScoreThreshold, global_params_.NmsOverlapThreshold, global_params_.PfeOnnxFile.c_str(), global_params_.BackboneOnnxFile.c_str(), global_params_.InputPCDFile.c_str(), global_params_.OutputDetsFile.c_str());
}

void PointPillarsPipeline::LoadModelConfigs(const char* model_yaml_file)
{
    YAML::Node params = YAML::LoadFile(model_yaml_file);
    model_params_.kPillarXSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][0].as<float>();
    model_params_.kPillarYSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][1].as<float>();
    model_params_.kPillarZSize = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["VOXEL_SIZE"][2].as<float>();
    model_params_.kMinXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][0].as<float>();
    model_params_.kMinYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][1].as<float>();
    model_params_.kMinZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][2].as<float>();
    model_params_.kMaxXRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][3].as<float>();
    model_params_.kMaxYRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][4].as<float>();
    model_params_.kMaxZRange = params["DATA_CONFIG"]["POINT_CLOUD_RANGE"][5].as<float>();
    model_params_.kNumClass = params["CLASS_NAMES"].size();
    model_params_.kMaxNumPillars = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_NUMBER_OF_VOXELS"]["test"].as<int>();
    model_params_.kMaxNumPointsPerPillar = params["DATA_CONFIG"]["DATA_PROCESSOR"][2]["MAX_POINTS_PER_VOXEL"].as<int>();
    model_params_.kNumPointFeature = 5; // 5 [x, y, z, i, r=0]
    model_params_.kNumGatherPointFeature = 11;
    model_params_.kNumInputBoxFeature = 7;
    model_params_.kNumOutputBoxFeature = params["MODEL"]["DENSE_HEAD"]["TARGET_ASSIGNER_CONFIG"]["BOX_CODER_CONFIG"]["code_size"].as<int>();
    model_params_.kBatchSize = 1;
    model_params_.kNumIndsForScan = 1024;
    model_params_.kNumThreads = 8;
    model_params_.kNumBoxCorners = 8;
    model_params_.kAnchorStrides = 4;
    model_params_.kNmsPreMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_PRE_MAXSIZE"].as<int>();
    model_params_.kNmsPostMaxsize = params["MODEL"]["POST_PROCESSING"]["NMS_CONFIG"]["NMS_POST_MAXSIZE"].as<int>();
    //params for initialize anchors
    //Adapt to OpenPCDet
    model_params_.kAnchorNames = params["CLASS_NAMES"].as<std::vector<std::string>>();
    for (size_t i = 0; i < model_params_.kAnchorNames.size(); ++i)
    {
        model_params_.kAnchorDxSizes.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_sizes"][0][0].as<float>());
        model_params_.kAnchorDySizes.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_sizes"][0][1].as<float>());
        model_params_.kAnchorDzSizes.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_sizes"][0][2].as<float>());
        model_params_.kAnchorBottom.emplace_back(params["MODEL"]["DENSE_HEAD"]["ANCHOR_GENERATOR_CONFIG"][i]["anchor_bottom_heights"][0].as<float>());
    }
    for (size_t idx_head = 0; idx_head < params["MODEL"]["DENSE_HEAD"]["RPN_HEAD_CFGS"].size(); ++idx_head)
    {
        int num_cls_per_head = params["MODEL"]["DENSE_HEAD"]["RPN_HEAD_CFGS"][idx_head]["HEAD_CLS_NAME"].size();
        std::vector<int> value;
        for (int i = 0; i < num_cls_per_head; ++i)
        {
            value.emplace_back(idx_head + i);
        }
        model_params_.kMultiheadLabelMapping.emplace_back(value);
    }

    // Generate secondary parameters based on above.
    model_params_.kGridXSize = static_cast<int>((model_params_.kMaxXRange - model_params_.kMinXRange) / model_params_.kPillarXSize); //512
    model_params_.kGridYSize = static_cast<int>((model_params_.kMaxYRange - model_params_.kMinYRange) / model_params_.kPillarYSize); //512
    model_params_.kGridZSize = static_cast<int>((model_params_.kMaxZRange - model_params_.kMinZRange) / model_params_.kPillarZSize); //1
    model_params_.kRpnInputSize = 64 * model_params_.kGridYSize * model_params_.kGridXSize;

    model_params_.kNumAnchorXinds = static_cast<int>(model_params_.kGridXSize / model_params_.kAnchorStrides); //Width
    model_params_.kNumAnchorYinds = static_cast<int>(model_params_.kGridYSize / model_params_.kAnchorStrides); //Hight
    model_params_.kNumAnchor = model_params_.kNumAnchorXinds * model_params_.kNumAnchorYinds * 2 * model_params_.kNumClass;  // H * W * Ro * N = 196608

    model_params_.kNumAnchorPerCls = model_params_.kNumAnchorXinds * model_params_.kNumAnchorYinds * 2; //H * W * Ro = 32768
    model_params_.kRpnBoxOutputSize = model_params_.kNumAnchor * model_params_.kNumOutputBoxFeature;
    model_params_.kRpnClsOutputSize = model_params_.kNumAnchor * model_params_.kNumClass;
    model_params_.kRpnDirOutputSize = model_params_.kNumAnchor * 2;
}

bool PointPillarsPipeline::DoPreProc()
{
    return true;
}

bool PointPillarsPipeline::InferPfeOnnxModel()
{
    return true;
}

bool PointPillarsPipeline::InferBackboneOnnxModel()
{
    return true;
}

bool PointPillarsPipeline::DoScatter()
{
    return true;
}

bool PointPillarsPipeline::DoPostProc()
{
    return true;
}

void PointPillarsPipeline::AllocBuffers()
{
    dev_num_points_per_pillar_ = new float[model_params_.kMaxNumPillars];
    dev_x_coors_ = new int[model_params_.kMaxNumPillars];
    dev_y_coors_ = new int[model_params_.kMaxNumPillars];
    dev_pillar_point_feature_ = new float[model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumPointFeature];
    dev_pillar_coors_ = new float[model_params_.kMaxNumPillars * 4 ];
    dev_sparse_pillar_map_ = new int[model_params_.kNumIndsForScan * model_params_.kNumIndsForScan];
    dev_cumsum_along_x_ = new int[model_params_.kNumIndsForScan * model_params_.kNumIndsForScan];
    dev_cumsum_along_y_ = new int[model_params_.kNumIndsForScan * model_params_.kNumIndsForScan];

    dev_pfe_gather_feature_ = new float[model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumGatherPointFeature];
    pfe_buffers_[0] = new float[model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumGatherPointFeature];
    pfe_buffers_[1] = new float[model_params_.kMaxNumPillars * 64];
    rpn_buffers_[0] = new float[model_params_.kRpnInputSize];
    rpn_buffers_[1] = new float[model_params_.kNumAnchorPerCls];
    rpn_buffers_[2] = new float[model_params_.kNumAnchorPerCls * 2 * 2];
    rpn_buffers_[3] = new float[model_params_.kNumAnchorPerCls * 2 * 2];
    rpn_buffers_[4] = new float[model_params_.kNumAnchorPerCls];
    rpn_buffers_[5] = new float[model_params_.kNumAnchorPerCls * 2 * 2];
    rpn_buffers_[6] = new float[model_params_.kNumAnchorPerCls * 2 * 2];
    rpn_buffers_[7] = new float[model_params_.kNumAnchorPerCls * model_params_.kNumClass * model_params_.kNumOutputBoxFeature];

    dev_scattered_feature_ = new float[model_params_.kNumThreads * model_params_.kGridYSize * model_params_.kGridXSize];
    host_box_ =  new float[model_params_.kNumAnchorPerCls * model_params_.kNumClass * model_params_.kNumOutputBoxFeature];
    host_score_ =  new float[model_params_.kNumAnchorPerCls * 18];
    host_filtered_count_ = new int[model_params_.kNumClass];
}

void PointPillarsPipeline::ClearBuffers()
{
    memset(dev_num_points_per_pillar_, 0, model_params_.kMaxNumPillars * sizeof(float));
    memset(dev_x_coors_, 0, model_params_.kMaxNumPillars * sizeof(int));
    memset(dev_y_coors_, 0, model_params_.kMaxNumPillars * sizeof(int));
    memset(dev_pillar_point_feature_, 0, model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumPointFeature * sizeof(float));
    memset(dev_pillar_coors_, 0, model_params_.kMaxNumPillars * 4 * sizeof(float));
    memset(dev_sparse_pillar_map_, 0, model_params_.kNumIndsForScan * model_params_.kNumIndsForScan * sizeof(int));
    memset(dev_pfe_gather_feature_, 0, model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumGatherPointFeature * sizeof(float));
    memset(pfe_buffers_[0], 0, model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumGatherPointFeature * sizeof(float));
    memset(pfe_buffers_[1], 0, model_params_.kMaxNumPillars * 64 * sizeof(float));
    memset(dev_scattered_feature_, 0, model_params_.kNumThreads * model_params_.kGridYSize * model_params_.kGridXSize * sizeof(float));
}

void PointPillarsPipeline::FreeBuffers()
{
    delete[] dev_num_points_per_pillar_;
    delete[] dev_x_coors_;
    delete[] dev_y_coors_;
    delete[] dev_pillar_point_feature_;
    delete[] dev_pillar_coors_;
    delete[] dev_sparse_pillar_map_;
    delete[] dev_cumsum_along_x_;
    delete[] dev_cumsum_along_y_;
    delete[] dev_pfe_gather_feature_;
    delete[] pfe_buffers_[0];
    delete[] pfe_buffers_[1];
    delete[] rpn_buffers_[0];
    delete[] rpn_buffers_[1];
    delete[] rpn_buffers_[2];
    delete[] rpn_buffers_[3];
    delete[] rpn_buffers_[4];
    delete[] rpn_buffers_[5];
    delete[] rpn_buffers_[6];
    delete[] rpn_buffers_[7];
    delete[] dev_scattered_feature_;
    delete[] host_box_;
    delete[] host_score_;
    delete[] host_filtered_count_;
}
