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