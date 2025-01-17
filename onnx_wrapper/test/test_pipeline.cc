#include <gtest/gtest.h>
#include "pointpillars_pipeline.h"
#include "crc_checker.h"

#define     GLOBAL_CONFIG_PATH      "bootstrap.yaml"
#define     MODEL_CONFIG_PATH       "pointpillars/cfgs/cbgs_pp_multihead.yaml"

class TestPointpillarsPipeline : public testing::Test, public PointPillarsPipeline
{
public:
    float* points_array_;
    int num_points_;
protected:
    virtual void SetUp()
    {
        RLOGI("TestPointpillarsPipeline SetUp");
        EXPECT_TRUE(Init(GLOBAL_CONFIG_PATH, MODEL_CONFIG_PATH));
        /** load file testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
        num_points_ = LoadPCDFile(global_params_.InputPCDFile.c_str(), 5);
        EXPECT_EQ(num_points_, 34720);
        points_array_ = this->pcd_array_.get();
    }

    virtual void TearDown()
    {
        RLOGI("TestPointpillarsPipeline TearDown");
        EXPECT_TRUE(Stop());
    }
};

TEST_F(TestPointpillarsPipeline, MakePillarHistoST)
{
    int32_t dev_pillar_count_histo_crc = 0;
    int32_t dev_pillar_point_feature_in_coors_crc = 0;

    bool ret = this->preproc_op_->TestMakePillarHistoST(
        dev_pillar_count_histo_crc,
        dev_pillar_point_feature_in_coors_crc,
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);
    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    EXPECT_EQ(dev_pillar_count_histo_crc, 0x19865099);
    EXPECT_EQ(dev_pillar_point_feature_in_coors_crc, 0x646ddb6);
}

TEST_F(TestPointpillarsPipeline, MakePillarIndexST)
{
    int32_t dev_counter_val = 0;
    int32_t dev_pillar_count_val = 0;
    int32_t dev_x_coors_crc = 0;
    int32_t dev_y_coors_crc = 0;
    int32_t dev_num_points_per_pillar_crc = 0;
    int32_t dev_sparse_pillar_map_crc = 0;

    bool ret = this->preproc_op_->TestMakePillarIndexST(
        dev_counter_val,
        dev_pillar_count_val,
        dev_x_coors_crc,
        dev_y_coors_crc,
        dev_num_points_per_pillar_crc,
        dev_sparse_pillar_map_crc,
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);
    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    EXPECT_EQ(dev_counter_val, 8697);
    EXPECT_EQ(dev_pillar_count_val, 8697);
    EXPECT_EQ(dev_x_coors_crc, 0x3e0653f2);
    EXPECT_EQ(dev_y_coors_crc, 0x81b64d6b);
    EXPECT_EQ(dev_num_points_per_pillar_crc, 0x9c9fbc7d);
    EXPECT_EQ(dev_sparse_pillar_map_crc, 0xf5930901);
}

TEST_F(TestPointpillarsPipeline, MakePillarFeatureST)
{
    int32_t dev_pillar_point_feature_crc = 0;
    int32_t dev_pillar_coors_crc = 0;

    bool ret = this->preproc_op_->TestMakePillarFeatureST(
        dev_pillar_point_feature_crc,
        dev_pillar_coors_crc,
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);
    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    EXPECT_EQ(dev_pillar_point_feature_crc, 0xd4929558);
    EXPECT_EQ(dev_pillar_coors_crc, 0x9605ba67);
}

TEST_F(TestPointpillarsPipeline, CalcPillarMeanST)
{
    int32_t dev_points_mean_crc = 0;

    bool ret = this->preproc_op_->TestCalcPillarMeanST(
        dev_points_mean_crc,
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);
    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    EXPECT_EQ(dev_points_mean_crc, 0x4a89e55f);
}

TEST_F(TestPointpillarsPipeline, RunPreProc)
{
    bool ret = this->preproc_op_->RunPreProc(
        dev_x_coors_, 
        dev_y_coors_,
        dev_num_points_per_pillar_, 
        dev_pillar_point_feature_, 
        dev_pillar_coors_,
        dev_sparse_pillar_map_, 
        host_pillar_count_ ,
        dev_pfe_gather_feature_,
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);
    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    int32_t dev_pfe_gather_feature_crc = gfCalcFloatsCRC(dev_pfe_gather_feature_, model_params_.kMaxNumPillars * model_params_.kMaxNumPointsPerPillar * model_params_.kNumGatherPointFeature, 6);
    LOGPF("dev_pfe_gather_feature_crc: 0x%x", dev_pfe_gather_feature_crc);
    EXPECT_EQ(dev_pfe_gather_feature_crc, 0x572adfff);
}

TEST_F(TestPointpillarsPipeline, InferPfeOnnxModel)
{
    bool ret = DoPreProc(
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);

    EXPECT_TRUE(LoadModels());

    ret = InferPfeOnnxModel();
    EXPECT_TRUE(ret);
    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    int32_t pfe_buffers_1_crc = gfCalcFloatsCRC(pfe_buffers_[1], model_params_.kMaxNumPillars * 64, 6);
    LOGPF("pfe_buffers_1_crc: 0x%x", pfe_buffers_1_crc);
    EXPECT_EQ(pfe_buffers_1_crc, 0xeca540e2);
}

TEST_F(TestPointpillarsPipeline, DoScatter)
{
    bool ret = DoPreProc(
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);

    EXPECT_TRUE(LoadModels());

    ret = InferPfeOnnxModel();
    EXPECT_TRUE(ret);

    ret = DoScatter();
    EXPECT_TRUE(ret);

    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    int32_t dev_scattered_feature_crc = gfCalcFloatsCRC(dev_scattered_feature_, model_params_.kNumThreads * model_params_.kGridYSize * model_params_.kGridXSize, 6);
    LOGPF("dev_scattered_feature_crc: 0x%x", dev_scattered_feature_crc);
    EXPECT_EQ(dev_scattered_feature_crc, 0x2c386bd4);
}

TEST_F(TestPointpillarsPipeline, InferBackboneOnnxModel)
{
    bool ret = DoPreProc(
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);

    EXPECT_TRUE(LoadModels());

    ret = InferPfeOnnxModel();
    EXPECT_TRUE(ret);

    ret = DoScatter();
    EXPECT_TRUE(ret);

    ret = InferBackboneOnnxModel();
    EXPECT_TRUE(ret);

    /** Tensor CRC for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    int32_t host_box_crc = gfCalcFloatsCRC(host_box_, model_params_.kNumAnchorPerCls * model_params_.kNumClass * model_params_.kNumOutputBoxFeature, 6);
    LOGPF("host_box_crc: 0x%x", host_box_crc);
    EXPECT_EQ(host_box_crc, 0x6fa6b143);
}

TEST_F(TestPointpillarsPipeline, RunPipeline)
{
    EXPECT_TRUE(LoadModels());

    std::vector<POINTPILLARS_BBOX3D_t> bboxes;
    bool ret = RunPipeline(
        bboxes,
        points_array_,
        num_points_
    );
    EXPECT_TRUE(ret);
    RLOGI("total bboxes: %ld", bboxes.size());
    /** output bboxes for testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt */
    EXPECT_EQ(bboxes.size(), 15);
    for(size_t i=0; i<bboxes.size(); i++)
    {
        const POINTPILLARS_BBOX3D_t& box_i = bboxes[i];
        RLOGI("box[%d]: xyz = (%.2f, %.2f, %.2f), wlh = (%.2f, %.2f, %.2f), yaw = %.2f, cls = %d, score = %.2f",\
               i, box_i.x, box_i.y, box_i.z, box_i.dx, box_i.dy, box_i.dz, box_i.yaw, box_i.cls, box_i.score);
    }
    PointpillarsOpsPostProc::SaveBox3dToFile(global_params_.OutputDetsFile, bboxes);
}