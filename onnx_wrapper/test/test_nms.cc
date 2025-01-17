#include <gtest/gtest.h>
#include "nms.h"
#include "crc_checker.h"


class TestPointpillarsOpsNMS : public testing::Test
{
public:
    constexpr static float NMS_TH_ = 0.2f;
    constexpr static int BOX_NUM_ = 1024;
    std::shared_ptr<PointpillarsOpsNMS> nms_op_;
protected:
    virtual void SetUp()
    {
        RLOGI("TestPointpillarsOpsNMS SetUp");
        nms_op_ = std::make_shared<PointpillarsOpsNMS>(64, 8, NMS_TH_);
    }

    virtual void TearDown()
    {
        RLOGI("TestPointpillarsOpsNMS TearDown");
    }
};

TEST_F(TestPointpillarsOpsNMS, DoNmsST_Diagonal)
{
    /** dim=7 box: float[x, y, z, dx, dy, dz, heading] */
    std::shared_ptr<float> in_boxes(new float[BOX_NUM_*7], [](float *p) { delete[] p; });
    for(int i=0; i<BOX_NUM_; i++)
    {
        float* box_i = in_boxes.get() + i * 7;
        box_i[0] = i;
        box_i[1] = i;
        box_i[2] = rand() % BOX_NUM_; // z is ignored in iou_bev()
        box_i[3] = 2.5f;
        box_i[4] = 2.5f;
        box_i[5] = rand() % BOX_NUM_;
        box_i[6] = 0.0f;
    }

    int out_num_to_keep = 0;
    long out_keep_inds[BOX_NUM_] = {0};

    bool ret = nms_op_ -> DoNmsST(
        &out_num_to_keep,
        out_keep_inds,
        BOX_NUM_,
        in_boxes.get()
    );

    EXPECT_TRUE(ret);
    RLOGI("out_num_to_keep: %d", out_num_to_keep);
    EXPECT_EQ(out_num_to_keep, BOX_NUM_/2);
}