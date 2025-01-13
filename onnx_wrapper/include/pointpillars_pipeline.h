#ifndef __POINTPILLARS_PIPELINE_H__
#define __POINTPILLARS_PIPELINE_H__

#include "ort_pointpillars_pfe_infer.h"
#include "ort_pointpillars_backbone_infer.h"
#include "nms.h"
#include "preproc.h"
#include "postproc.h"
#include "scatter.h"

typedef struct 
{
    int BoxFeature;
    float ScoreThreshold;
    float NmsOverlapThreshold;
    std::string PfeOnnxFile;
    std::string BackboneOnnxFile;
    std::string InputPCDFile;
    std::string OutputDetsFile; 
} PointpillarsGlobalConfigs_t;

typedef struct 
{
    // voxel size
    float kPillarXSize;
    float kPillarYSize;
    float kPillarZSize;
    // point cloud range
    float kMinXRange;
    float kMinYRange;
    float kMinZRange;
    float kMaxXRange;
    float kMaxYRange;
    float kMaxZRange;
    // hyper parameters
    int kNumClass;
    int kMaxNumPillars;
    int kMaxNumPointsPerPillar;
    int kNumPointFeature;
    int kNumGatherPointFeature;
    int kGridXSize;
    int kGridYSize;
    int kGridZSize;
    int kNumAnchorXinds;
    int kNumAnchorYinds;
    int kRpnInputSize;
    int kNumAnchor;
    int kNumInputBoxFeature;
    int kNumOutputBoxFeature;
    int kRpnBoxOutputSize;
    int kRpnClsOutputSize;
    int kRpnDirOutputSize;
    int kBatchSize;
    int kNumIndsForScan;
    int kNumThreads;
    // if you change kNumThreads, need to modify NUM_THREADS_MACRO in
    // common.h
    int kNumBoxCorners;
    int kNmsPreMaxsize;
    int kNmsPostMaxsize;
    //params for initialize anchors
    //Adapt to OpenPCDet
    int kAnchorStrides;
    std::vector<std::string> kAnchorNames;
    std::vector<float> kAnchorDxSizes;
    std::vector<float> kAnchorDySizes;
    std::vector<float> kAnchorDzSizes;
    std::vector<float> kAnchorBottom;
    std::vector<std::vector<int>> kMultiheadLabelMapping;
    int kNumAnchorPerCls;
} PointpillarsModelConfigs_t;

class PointPillarsPipeline
{
public:
    PointPillarsPipeline()
    {
    }

    ~PointPillarsPipeline()
    {
    }

    bool Init(const char* global_config_path, const char* model_config_path);
    bool Stop();
    void RunTest();
    bool RunPipeline();

protected:
    size_t LoadPCDFile(const char* pcd_txt_path, const int num_feature=5);
    void LoadGlobalConfigs(const char* global_yaml_file);
    void LoadModelConfigs(const char* model_yaml_file);

    void AllocBuffers();
    void ClearBuffers();
    void FreeBuffers();

    bool DoPreProc();
    bool InferPfeOnnxModel();
    bool InferBackboneOnnxModel();
    bool DoScatter();
    bool DoPostProc();

protected:
    std::shared_ptr<OrtPointPillarsPfeInfer> ort_pfe_model_;
    std::shared_ptr<OrtPointPillarsBackboneInfer> ort_backbone_model_;
    std::shared_ptr<float> pcd_array_;
    PointpillarsGlobalConfigs_t global_params_;
    PointpillarsModelConfigs_t model_params_;

    std::shared_ptr<PointpillarsOpsPreProc> preproc_op_;
    std::shared_ptr<PointpillarsOpsScatter> scatter_op_;
    std::shared_ptr<PointpillarsOpsPostProc> postproc_op_;
    std::shared_ptr<PointpillarsOpsNMS> nms_op_;

    int host_pillar_count_[1];
    int* dev_x_coors_;
    int* dev_y_coors_;
    float* dev_num_points_per_pillar_;
    int* dev_sparse_pillar_map_;
    int* dev_cumsum_along_x_;
    int* dev_cumsum_along_y_;

    float* dev_pillar_point_feature_;
    float* dev_pillar_coors_;
    float* dev_points_mean_;

    float* dev_pfe_gather_feature_;
    float* pfe_buffers_[2];
    float* rpn_buffers_[8];

    std::vector<float*> rpn_box_output_; 
    std::vector<float*> rpn_cls_output_;

    float* dev_scattered_feature_;

    float* host_box_;
    float* host_score_;
    int*   host_filtered_count_;

};

#endif //__POINTPILLARS_PIPELINE_H__