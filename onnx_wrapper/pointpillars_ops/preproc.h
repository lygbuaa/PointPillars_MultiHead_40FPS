#ifndef __POINTPILLARS_OPS_PREPROC_H__
#define __POINTPILLARS_OPS_PREPROC_H__

#include <cstring>
#include <cmath>
#include "logging_utils.h"

class PointpillarsOpsPreProc
{
public:
    PointpillarsOpsPreProc(
        const int num_threads,
        const int max_num_pillars,
        const int max_points_per_pillar,
        const int num_point_feature,
        const int num_inds_for_scan,
        const int grid_x_size,
        const int grid_y_size,
        const int grid_z_size,
        const float pillar_x_size,
        const float pillar_y_size,
        const float pillar_z_size,
        const float min_x_range, 
        const float min_y_range,
        const float min_z_range
    );
    ~PointpillarsOpsPreProc();

    void ClearBuffers();

    /**
     * @brief  RunPreProc
     *         process input points, output dim=11 feats
     *
     * @param[out]  dev_x_coors                         pillar x coor
     * @param[out]  dev_y_coors                         pillar y coor
     * @param[out]  dev_num_points_per_pillar           dense pillar map, float[kMaxNumPillars]
     * @param[out]  dev_pillar_point_feature            feats arranged according to pillar index
     * @param[out]  dev_pillar_coors                    pillar coord fusion, float[kMaxNumPillars * 4]
     * @param[out]  dev_sparse_pillar_map               sparse pillar map, int[1024*1024]
     * @param[out]  host_pillar_count                   pillar count
     * @param[out]  dev_pfe_gather_feature              gathered dim=11 feats, float[kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature]
     * @param[in]   dev_points                          input points, dim=5, x,y,z,i,r
     * @param[in]   in_num_points                       number of input points
     *
     * @return
     *
     * @retval true     success
     * @retval false    failed
     */
    bool RunPreProc(
        int* const dev_x_coors,
        int* const dev_y_coors,
        float* const dev_num_points_per_pillar,
        float* const dev_pillar_point_feature,
        float* const dev_pillar_coors,
        int* const dev_sparse_pillar_map,
        int* const host_pillar_count,
        float* const dev_pfe_gather_feature,
        const float* dev_points,
        const int in_num_points
    );

    /**
     * @brief  MakePillarHistoST
     *         make histogram pillars from points, single thread
     *
     * @param[out]  dev_pillar_point_feature_in_coors   feats arranged according to pillar coord index, float[grid_y_size*grid_x_size*max_num_points_per_pillar*num_point_feature]
     * @param[out]  pillar_count_histo                  points count in each pillar, int[grid_y_size*grid_x_size]
     * @param[in]   dev_points                          points, dim=5: x,y,z,i,r
     * @param[in]   num_points                          number of points
     * @param[in]   max_points_per_pillar               MAX_POINTS_PER_VOXEL: 20
     * @param[in]   grid_x_size                         ((kMaxXRange - kMinXRange) / kPillarXSize) = ((51.2f - (-51.2f) / 0.2f)) = 512
     * @param[in]   grid_y_size                         ((kMaxYRange - kMinYRange) / kPillarYSize) = ((51.2f - (-51.2f) / 0.2f)) = 512
     * @param[in]   grid_z_size                         ((kMaxZRange - kMinZRange) / kPillarZSize) = ((3.0f - (-5.0f) / 8.0f)) = 1
     * @param[in]   min_x_range                         kMinXRange = -51.2f
     * @param[in]   min_y_range                         kMinYRange = -51.2f
     * @param[in]   min_z_range                         kMinZRange = -5.0f
     * @param[in]   pillar_x_size                       VOXEL_SIZE: [0.2, 0.2, 8.0]
     * @param[in]   pillar_y_size                       VOXEL_SIZE: [0.2, 0.2, 8.0]
     * @param[in]   pillar_z_size                       VOXEL_SIZE: [0.2, 0.2, 8.0]
     * @param[in]   num_point_feature                   point_feature_dim = 5: x,y,z,i,r
     *
     * @return
     *
     * @retval true     success
     * @retval false    failed
     */
    bool MakePillarHistoST(
        float* const dev_pillar_point_feature_in_coors,
        int* const pillar_count_histo,
        const float* dev_points,
        const int num_points,
        const int max_points_per_pillar,
        const int grid_x_size,
        const int grid_y_size,
        const int grid_z_size,
        const float min_x_range,
        const float min_y_range, 
        const float min_z_range, 
        const float pillar_x_size,
        const float pillar_y_size, 
        const float pillar_z_size,
        const int num_point_feature
    );

    /**
     * @brief  MakePillarIndexST
     *         make histogram pillars from points, single thread
     *
     * @param[out]  dev_counter                         counter for non-empty pillars
     * @param[out]  dev_pillar_count                    counter for non-empty && non-full pillars
     * @param[out]  dev_x_coors                         pillar x coor
     * @param[out]  dev_y_coors                         pillar y coor
     * @param[out]  dev_num_points_per_pillar           dense pillar map, float[kMaxNumPillars]
     * @param[out]  dev_sparse_pillar_map               sparse pillar map, int[1024*1024]
     * @param[in]   dev_pillar_count_histo              histogram from MakePillarHistoST()
     * @param[in]   max_pillars                         kMaxNumPillars, 30000
     * @param[in]   max_points_per_pillar               MAX_POINTS_PER_VOXEL: 20
     * @param[in]   grid_x_size                         512
     * @param[in]   num_inds_for_scan                   kNumIndsForScan, 1024
     *
     * @return
     *
     * @retval true     success
     * @retval false    failed
     */
    bool MakePillarIndexST(
        int* const dev_counter,
        int* const dev_pillar_count,
        int* const dev_x_coors,
        int* const dev_y_coors,
        float* const dev_num_points_per_pillar,
        int* const dev_sparse_pillar_map,
        const int* dev_pillar_count_histo,
        const int max_pillars,
        const int max_points_per_pillar, 
        const int grid_x_size,
        const int num_inds_for_scan
    );

    /**
     * @brief  MakePillarFeatureST
     *         make pillar features from points
     *
     * @param[out]  dev_pillar_point_feature                feats arranged according to pillar index, that is (512*512-->30000), float[kMaxNumPillars * kMaxNumPointsPerPillar * kNumPointFeature]
     * @param[out]  dev_pillar_coors                        pillar coord fusion, float[kMaxNumPillars * 4]
     * @param[in]   dev_pillar_point_feature_in_coors       feats arranged according to pillar coord index, comes from MakePillarHistoST(), float[grid_y_size*grid_x_size*max_num_points_per_pillar*num_point_feature]
     * @param[in]   dev_x_coors                             pillar x coor, comes from MakePillarIndexST()
     * @param[in]   dev_y_coors                             pillar y coor, comes from MakePillarIndexST()
     * @param[in]   dev_num_points_per_pillar               dense pillar map, comes from MakePillarIndexST()
     * @param[in]   max_points_per_pillar                   MAX_POINTS_PER_VOXEL: 20
     * @param[in]   num_point_feature                       point_feature_dim = 5: x,y,z,i,r
     * @param[in]   grid_x_size                             512
     *
     * @return
     *
     * @retval true     success
     * @retval false    failed
     */
    bool MakePillarFeatureST(
        float* const dev_pillar_point_feature,
        float* const dev_pillar_coors,
        const float* dev_pillar_point_feature_in_coors,
        const int* dev_x_coors,
        const int* dev_y_coors,
        const float* dev_num_points_per_pillar,
        const int max_points_per_pillar,
        const int num_point_feature,
        const int grid_x_size
    );

    /**
     * @brief  CalcPillarMeanST
     *         calc mean value of (x,y,z) within each pillar
     *
     * @param[out]  dev_points_mean                         points (x,y,z) mean val within each pillar, float[max_num_pillars_ * 3]
     * @param[in]   num_point_feature                       point_feature_dim = 5: x,y,z,i,r
     * @param[in]   dev_pillar_point_feature                feats arranged according to pillar index, from MakePillarFeatureST(), float[kMaxNumPillars * kMaxNumPointsPerPillar * kNumPointFeature]
     * @param[in]   dev_num_points_per_pillar               dense pillar map, comes from MakePillarIndexST()
     * @param[in]   max_pillars                             reserved, kMaxNumPillars = 30000
     * @param[in]   max_points_per_pillar                   MAX_POINTS_PER_VOXEL: 20
     *
     * @return
     *
     * @retval true     success
     * @retval false    failed
     */
    bool CalcPillarMeanST(
        float* const dev_points_mean,
        const int num_point_feature,
        const float* dev_pillar_point_feature,
        const float* dev_num_points_per_pillar,
        const int max_pillars,
        const int max_points_per_pillar
    );

    /**
     * @brief  GatherPointFeatureST
     *         gather all the dim=11 features as PFE model input
     *
     * @param[out]  dev_pfe_gather_feature                  gathered dim=11 feats, float[kMaxNumPillars * kMaxNumPointsPerPillar * kNumGatherPointFeature]
     * @param[in]   max_num_pillars                         kMaxNumPillars = 30000
     * @param[in]   max_num_points_per_pillar               MAX_POINTS_PER_VOXEL: 20
     * @param[in]   num_point_feature                       point_feature_dim = 5: x,y,z,i,r
     * @param[in]   min_x_range                             kMinXRange = -51.2f
     * @param[in]   min_y_range                             kMinYRange = -51.2f
     * @param[in]   min_z_range                             kMinZRange = -5.0f
     * @param[in]   pillar_x_size                           VOXEL_SIZE: [0.2, 0.2, 8.0]
     * @param[in]   pillar_y_size                           VOXEL_SIZE: [0.2, 0.2, 8.0]
     * @param[in]   pillar_z_size                           VOXEL_SIZE: [0.2, 0.2, 8.0]
     * @param[in]   dev_pillar_point_feature                feats arranged according to pillar index, from MakePillarFeatureST(),
     * @param[in]   dev_num_points_per_pillar               dense pillar map, comes from MakePillarIndexST()
     * @param[in]   dev_pillar_coors                        pillar coord fusion, comes from MakePillarFeatureST()
     * @param[in]   dev_points_mean                         points (x,y,z) mean val within each pillar, comes from CalcPillarMeanST()
     *
     * @return
     *
     * @retval true     success
     * @retval false    failed
     */
    bool GatherPointFeatureST(
        float* const dev_pfe_gather_feature,
        const int max_num_pillars,
        const int max_num_points_per_pillar,
        const int num_point_feature,
        const float min_x_range, 
        const float min_y_range,
        const float min_z_range, 
        const float pillar_x_size,
        const float pillar_y_size,
        const float pillar_z_size,
        const float* dev_pillar_point_feature,
        const float* dev_num_points_per_pillar, 
        const float* dev_pillar_coors,
        const float* dev_points_mean
    );

    bool TestMakePillarHistoST(
        int32_t& dev_pillar_count_histo_crc,
        int32_t& dev_pillar_point_feature_in_coors_sum,
        const float* points_ptr, 
        const int num_points
    );

    bool TestMakePillarIndexST(
        int32_t& dev_counter_val,
        int32_t& dev_pillar_count_val,
        int32_t& dev_x_coors_crc,
        int32_t& dev_y_coors_crc,
        int32_t& dev_num_points_per_pillar_crc,
        int32_t& dev_sparse_pillar_map_crc,
        const float* points_ptr, 
        const int num_points
    );

    bool TestMakePillarFeatureST(
        int32_t& dev_pillar_point_feature_crc,
        int32_t& dev_pillar_coors_crc,
        const float* points_ptr,
        const int num_points
    );

    bool TestCalcPillarMeanST(
        int32_t& dev_points_mean_crc,
        const float* points_ptr, 
        const int num_points
    );

private:
    static constexpr int kNumGatherPointFeature_ = 11;
    const int num_threads_;
    const int max_num_pillars_;
    const int max_num_points_per_pillar_;
    const int num_point_feature_;
    const int num_inds_for_scan_;
    const int grid_x_size_;
    const int grid_y_size_;
    const int grid_z_size_;
    const float pillar_x_size_;
    const float pillar_y_size_;
    const float pillar_z_size_;
    const float min_x_range_;
    const float min_y_range_;
    const float min_z_range_;

    float* dev_pillar_point_feature_in_coors_;
    int* dev_pillar_count_histo_;

    int* dev_counter_;
    int* dev_pillar_count_;
    float* dev_points_mean_;

};

#endif