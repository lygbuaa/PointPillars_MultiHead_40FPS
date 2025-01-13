#include "crc_checker.h"
#include "preproc.h"

PointpillarsOpsPreProc::PointpillarsOpsPreProc(
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
)
:   num_threads_(num_threads),
    max_num_pillars_(max_num_pillars),
    max_num_points_per_pillar_(max_points_per_pillar),
    num_point_feature_(num_point_feature),
    num_inds_for_scan_(num_inds_for_scan),
    grid_x_size_(grid_x_size),
    grid_y_size_(grid_y_size),
    grid_z_size_(grid_z_size),
    pillar_x_size_(pillar_x_size),
    pillar_y_size_(pillar_y_size),
    pillar_z_size_(pillar_z_size),
    min_x_range_(min_x_range),
    min_y_range_(min_y_range),
    min_z_range_(min_z_range)
{
    dev_pillar_point_feature_in_coors_ = new float[grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * num_point_feature_];
    dev_pillar_count_histo_ = new int[grid_y_size_ * grid_x_size_];
    dev_counter_ = new int;
    dev_pillar_count_ = new int;
    dev_points_mean_ = new float[max_num_pillars_ * 3];
    ClearBuffers();

    RLOGI("PointpillarsOpsPreProc init. num_threads_: %d, max_num_pillars_: %d, max_num_points_per_pillar_: %d, num_point_feature_: %d, num_inds_for_scan_: %d,\
           grid_x_size_: %d, grid_y_size_: %d, grid_z_size_: %d, pillar_x_size_: %.2f, pillar_y_size_: %.2f, pillar_z_size_: %.2f, min_x_range_: %.2f, min_y_range_: %.2f, min_z_range_: %.2f",\
           num_threads_, max_num_pillars_, max_num_points_per_pillar_, num_point_feature_, num_inds_for_scan_, \
           grid_x_size_, grid_y_size_, grid_z_size_, pillar_x_size_, pillar_y_size_, pillar_z_size_, min_x_range_, min_y_range_, min_z_range_
    );
}

PointpillarsOpsPreProc::~PointpillarsOpsPreProc()
{
    delete[] dev_pillar_point_feature_in_coors_;
    delete[] dev_pillar_count_histo_;
    delete dev_counter_;
    delete dev_pillar_count_;
    delete[] dev_points_mean_;
}

void PointpillarsOpsPreProc::ClearBuffers()
{
    memset(dev_pillar_point_feature_in_coors_, 0, grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * num_point_feature_ * sizeof(float));
    memset(dev_pillar_count_histo_, 0, grid_y_size_ * grid_x_size_ * sizeof(int));
    memset(dev_counter_, 0, sizeof(int));
    memset(dev_pillar_count_, 0, sizeof(int));
    memset(dev_points_mean_, 0, max_num_pillars_ * 3 * sizeof(float));
}

bool PointpillarsOpsPreProc::RunPreProc(
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
)
{
    ClearBuffers();

    bool ret = MakePillarHistoST(
        dev_pillar_point_feature_in_coors_,
        dev_pillar_count_histo_,
        dev_points,
        in_num_points,
        max_num_points_per_pillar_, 
        grid_x_size_, 
        grid_y_size_,
        grid_z_size_, 
        min_x_range_, 
        min_y_range_, 
        min_z_range_, 
        pillar_x_size_,
        pillar_y_size_, 
        pillar_z_size_, 
        num_point_feature_
    );

    if(!ret)
    {
        return ret;
    }

    ret = MakePillarIndexST(
        dev_counter_,
        dev_pillar_count_,
        dev_x_coors,
        dev_y_coors,
        dev_num_points_per_pillar,
        dev_sparse_pillar_map,
        dev_pillar_count_histo_,
        max_num_pillars_,
        max_num_points_per_pillar_,
        grid_x_size_,
        num_inds_for_scan_
    );

    if(!ret)
    {
        return ret;
    }

    ret = MakePillarFeatureST(
        dev_pillar_point_feature,
        dev_pillar_coors,
        dev_pillar_point_feature_in_coors_,
        dev_x_coors,
        dev_y_coors,
        dev_num_points_per_pillar,
        max_num_points_per_pillar_,
        num_point_feature_,
        grid_x_size_
    );

    if(!ret)
    {
        return ret;
    }

    ret = CalcPillarMeanST(
        dev_points_mean_,
        num_point_feature_, 
        dev_pillar_point_feature, 
        dev_num_points_per_pillar, 
        max_num_pillars_,
        max_num_points_per_pillar_
    );

    if(!ret)
    {
        return ret;
    }

    ret = GatherPointFeatureST(
        dev_pfe_gather_feature,
        max_num_pillars_,
        max_num_points_per_pillar_,
        num_point_feature_,
        min_x_range_,
        min_y_range_,
        min_z_range_,
        pillar_x_size_,
        pillar_y_size_,
        pillar_z_size_, 
        dev_pillar_point_feature,
        dev_num_points_per_pillar,
        dev_pillar_coors,
        dev_points_mean_
    );

    if(!ret)
    {
        return ret;
    }

    return true;
}

bool PointpillarsOpsPreProc::MakePillarHistoST(
    float* dev_pillar_point_feature_in_coors,
    int* pillar_count_histo,
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
)
{
    for(int th_i=0; th_i<num_points; th_i++)
    {
        const int x_coor = floor((dev_points[th_i * num_point_feature + 0] - min_x_range) / pillar_x_size);
        const int y_coor = floor((dev_points[th_i * num_point_feature + 1] - min_y_range) / pillar_y_size);
        const int z_coor = floor((dev_points[th_i * num_point_feature + 2] - min_z_range) / pillar_z_size);

        if (x_coor >= 0 && x_coor < grid_x_size && 
            y_coor >= 0 && y_coor < grid_y_size && 
            z_coor >= 0 && z_coor < grid_z_size)
        {
            const int xy_coor = y_coor * grid_x_size + x_coor;
            /** atomicAdd() returns old value */
            const int count = pillar_count_histo[xy_coor];
            pillar_count_histo[xy_coor] += 1;

            if(count < max_points_per_pillar)
            {
                int ind = (y_coor * grid_x_size + x_coor) * max_points_per_pillar * num_point_feature + count * num_point_feature;
                for (int i = 0; i<num_point_feature; ++i)
                {
                    dev_pillar_point_feature_in_coors[ind + i] = dev_points[th_i * num_point_feature + i];
                }
            }
        }
    }
    return true;
}

bool PointpillarsOpsPreProc::MakePillarIndexST(
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
)
{
    for(int y=0; y<grid_y_size_; y++)
    {
        for(int x=0; x<grid_x_size_; x++)
        {
            const int num_points_at_this_pillar = dev_pillar_count_histo[y * grid_x_size + x];
            /** discard empty pillars */
            if(num_points_at_this_pillar < 1) 
            {
                continue;
            }
            /** atomicAdd() returns old value */
            const int count = *dev_counter;
            *dev_counter += 1;
            if(count < max_pillars)
            {
                *dev_pillar_count += 1;
                if(num_points_at_this_pillar >= max_points_per_pillar)
                {
                    dev_num_points_per_pillar[count] = max_points_per_pillar;
                }
                else
                {
                    dev_num_points_per_pillar[count] = num_points_at_this_pillar;
                }
                dev_x_coors[count] = x;
                dev_y_coors[count] = y;
                dev_sparse_pillar_map[y * num_inds_for_scan + x] = 1;
            }
        }
    }

    return true;
}

bool PointpillarsOpsPreProc::MakePillarFeatureST(
    float* const dev_pillar_point_feature,
    float* const dev_pillar_coors,
    const float* dev_pillar_point_feature_in_coors,
    const int* dev_x_coors,
    const int* dev_y_coors,
    const float* dev_num_points_per_pillar,
    const int max_points_per_pillar,
    const int num_point_feature,
    const int grid_x_size
)
{
    for(int ith_pillar=0; ith_pillar<*dev_pillar_count_; ith_pillar++)
    {
        const int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
        const int x_ind = dev_x_coors[ith_pillar];
        const int y_ind = dev_y_coors[ith_pillar];

        for(int ith_point=0; ith_point<max_num_points_per_pillar_; ith_point++)
        {
            /** pillar overflow will not happen after MakePillarIndexST */
            if (ith_point >= num_points_at_this_pillar)
            {
                break;
            }
            const int pillar_ind = ith_pillar * max_points_per_pillar * num_point_feature + ith_point * num_point_feature;
            const int coors_ind = y_ind * grid_x_size * max_points_per_pillar * num_point_feature + x_ind * max_points_per_pillar * num_point_feature + ith_point * num_point_feature;

            for (int i = 0; i < num_point_feature; ++i)
            {
                dev_pillar_point_feature[pillar_ind + i] = dev_pillar_point_feature_in_coors[coors_ind + i];
            }
            dev_pillar_coors[ith_pillar * 4 + 0] = 0.0f;  // batch idx
            dev_pillar_coors[ith_pillar * 4 + 1] = 0.0f;  // z
            dev_pillar_coors[ith_pillar * 4 + 2] = static_cast<float>(y_ind);
            dev_pillar_coors[ith_pillar * 4 + 3] = static_cast<float>(x_ind);
        }
    }
    return true;
}

bool PointpillarsOpsPreProc::CalcPillarMeanST(
    float* const dev_points_mean,
    const int num_point_feature,
    const float* dev_pillar_point_feature,
    const float* dev_num_points_per_pillar,
    const int max_pillars,
    const int max_points_per_pillar
)
{
    for(int ith_pillar=0; ith_pillar<*dev_pillar_count_; ith_pillar++)
    {
        const int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
        float feat_sum[3] = {0.0f};
        for(int ith_point=0; ith_point<num_points_at_this_pillar; ith_point++)
        {
            for(int axis=0; axis<3; axis++)
            {
                const float& tmp_feat = dev_pillar_point_feature[ith_pillar * max_points_per_pillar * num_point_feature + ith_point * num_point_feature + axis];
                feat_sum[axis] += tmp_feat;
            }
        }

        for(int axis=0; axis<3; axis++)
        {
            dev_points_mean[ith_pillar * 3 + axis] = feat_sum[axis] / num_points_at_this_pillar ;
        }
    }
    return true;
}


// gather_point_feature_kernel<<<max_num_pillars_, max_num_points_per_pillar_>>>

bool PointpillarsOpsPreProc::GatherPointFeatureST(
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
)
{
    static constexpr int num_gather_feature = 11;
    for(int ith_pillar=0; ith_pillar<*dev_pillar_count_; ith_pillar++)
    {
        const int num_points_at_this_pillar = dev_num_points_per_pillar[ith_pillar];
        for(int ith_point=0; ith_point<num_points_at_this_pillar; ith_point++)
        {
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 0] 
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0]; 
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 1]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1];
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 2]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2];
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 3]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 3];

            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 4]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 4];
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 5]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0] - dev_points_mean[ith_pillar * 3 + 0 ];

            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 6] 
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1] - dev_points_mean[ith_pillar * 3 + 1 ];
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 7]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2] - dev_points_mean[ith_pillar * 3 + 2 ];

            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 8]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 0] - (dev_pillar_coors[ith_pillar * 4 + 3] * pillar_x_size + (pillar_x_size/2 + min_x_range));
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 9]  
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 1] - (dev_pillar_coors[ith_pillar * 4 + 2] * pillar_y_size + (pillar_y_size/2 + min_y_range));
        
            dev_pfe_gather_feature[ith_pillar * max_num_points_per_pillar * num_gather_feature + ith_point * num_gather_feature + 10] 
            = dev_pillar_point_feature[ith_pillar * max_num_points_per_pillar * num_point_feature + ith_point * num_point_feature + 2] - (dev_pillar_coors[ith_pillar * 4 + 1] * pillar_z_size + (pillar_z_size/2 + min_z_range));
        }
    }

    return true;
}

bool PointpillarsOpsPreProc::TestMakePillarHistoST(
    int32_t& dev_pillar_count_histo_crc,
    int32_t& dev_pillar_point_feature_in_coors_crc,
    const float* points_ptr, 
    const int num_points
)
{
    bool ret = MakePillarHistoST(
        dev_pillar_point_feature_in_coors_,
        dev_pillar_count_histo_,
        points_ptr,
        num_points,
        max_num_points_per_pillar_, 
        grid_x_size_, 
        grid_y_size_,
        grid_z_size_, 
        min_x_range_, 
        min_y_range_, 
        min_z_range_, 
        pillar_x_size_,
        pillar_y_size_, 
        pillar_z_size_, 
        num_point_feature_
    );

    dev_pillar_count_histo_crc = gfCalcBytesCRC(dev_pillar_count_histo_, grid_y_size_ * grid_x_size_ * sizeof(int));
    dev_pillar_point_feature_in_coors_crc = gfCalcFloatsCRC(dev_pillar_point_feature_in_coors_, grid_y_size_ * grid_x_size_ * max_num_points_per_pillar_ * num_point_feature_, 6);
    RLOGI("dev_pillar_count_histo_crc: 0x%x, dev_pillar_point_feature_in_coors_crc: 0x%x", dev_pillar_count_histo_crc, dev_pillar_point_feature_in_coors_crc);
    return ret;
}

bool PointpillarsOpsPreProc::TestMakePillarIndexST(
    int32_t& dev_counter_val,
    int32_t& dev_pillar_count_val,
    int32_t& dev_x_coors_crc,
    int32_t& dev_y_coors_crc,
    int32_t& dev_num_points_per_pillar_crc,
    int32_t& dev_sparse_pillar_map_crc,
    const float* points_ptr, 
    const int num_points
)
{
    bool ret = MakePillarHistoST(
        dev_pillar_point_feature_in_coors_,
        dev_pillar_count_histo_,
        points_ptr,
        num_points,
        max_num_points_per_pillar_, 
        grid_x_size_, 
        grid_y_size_,
        grid_z_size_, 
        min_x_range_, 
        min_y_range_, 
        min_z_range_, 
        pillar_x_size_,
        pillar_y_size_, 
        pillar_z_size_, 
        num_point_feature_
    );

    if(!ret)
    {
        return ret;
    }

    int* dev_x_coors_cpu = new int[max_num_pillars_];
    int* dev_y_coors_cpu = new int[max_num_pillars_];
    float* dev_num_points_per_pillar_cpu = new float[max_num_pillars_];
    int* dev_sparse_pillar_map_cpu = new int[num_inds_for_scan_*num_inds_for_scan_];
    memset(dev_x_coors_cpu, 0, max_num_pillars_*sizeof(int));
    memset(dev_y_coors_cpu, 0, max_num_pillars_*sizeof(int));
    memset(dev_num_points_per_pillar_cpu, 0, max_num_pillars_*sizeof(float));
    memset(dev_sparse_pillar_map_cpu, 0, num_inds_for_scan_*num_inds_for_scan_*sizeof(int));

    ret = MakePillarIndexST(
        dev_counter_,
        dev_pillar_count_,
        dev_x_coors_cpu,
        dev_y_coors_cpu,
        dev_num_points_per_pillar_cpu,
        dev_sparse_pillar_map_cpu,
        dev_pillar_count_histo_,
        max_num_pillars_,
        max_num_points_per_pillar_,
        grid_x_size_,
        num_inds_for_scan_
    );

    dev_counter_val = *dev_counter_;
    dev_pillar_count_val = *dev_pillar_count_;
    dev_x_coors_crc = gfCalcBytesCRC(dev_x_coors_cpu, max_num_pillars_*sizeof(int));
    dev_y_coors_crc = gfCalcBytesCRC(dev_y_coors_cpu, max_num_pillars_*sizeof(int));
    /** dev_num_points_per_pillar_cpu are ints */
    dev_num_points_per_pillar_crc = gfCalcFloatsCRC(dev_num_points_per_pillar_cpu, max_num_pillars_, 0);
    dev_sparse_pillar_map_crc = gfCalcBytesCRC(dev_sparse_pillar_map_cpu, num_inds_for_scan_*num_inds_for_scan_*sizeof(int));
    LOGPF("dev_counter_val: %d, dev_pillar_count_val: %d", dev_counter_val, dev_pillar_count_val);
    LOGPF("dev_x_coors_crc: 0x%x, dev_y_coors_crc: 0x%x, dev_num_points_per_pillar_crc: 0x%x, dev_sparse_pillar_map_crc: 0x%x",  \
           dev_x_coors_crc, dev_y_coors_crc, dev_num_points_per_pillar_crc, dev_sparse_pillar_map_crc);

    delete[] dev_x_coors_cpu;
    delete[] dev_y_coors_cpu;
    delete[] dev_num_points_per_pillar_cpu;
    delete[] dev_sparse_pillar_map_cpu;

    return ret;
}

bool PointpillarsOpsPreProc::TestMakePillarFeatureST(
    int32_t& dev_pillar_point_feature_crc,
    int32_t& dev_pillar_coors_crc,
    const float* points_ptr, 
    const int num_points
)
{
    bool ret = MakePillarHistoST(
        dev_pillar_point_feature_in_coors_,
        dev_pillar_count_histo_,
        points_ptr,
        num_points,
        max_num_points_per_pillar_, 
        grid_x_size_, 
        grid_y_size_,
        grid_z_size_, 
        min_x_range_, 
        min_y_range_, 
        min_z_range_, 
        pillar_x_size_,
        pillar_y_size_, 
        pillar_z_size_, 
        num_point_feature_
    );

    if(!ret)
    {
        return ret;
    }

    int* dev_x_coors_cpu = new int[max_num_pillars_];
    int* dev_y_coors_cpu = new int[max_num_pillars_];
    float* dev_num_points_per_pillar_cpu = new float[max_num_pillars_];
    int* dev_sparse_pillar_map_cpu = new int[num_inds_for_scan_*num_inds_for_scan_];
    memset(dev_x_coors_cpu, 0, max_num_pillars_*sizeof(int));
    memset(dev_y_coors_cpu, 0, max_num_pillars_*sizeof(int));
    memset(dev_num_points_per_pillar_cpu, 0, max_num_pillars_*sizeof(float));
    memset(dev_sparse_pillar_map_cpu, 0, num_inds_for_scan_*num_inds_for_scan_*sizeof(int));

    float* dev_pillar_point_feature_cpu = new float[max_num_pillars_*max_num_points_per_pillar_*num_point_feature_];
    float* dev_pillar_coors_cpu = new float[max_num_pillars_*4];
    memset(dev_pillar_point_feature_cpu, 0, max_num_pillars_*max_num_points_per_pillar_*num_point_feature_*sizeof(float));
    memset(dev_pillar_coors_cpu, 0, max_num_pillars_*4*sizeof(float));

    ret = MakePillarIndexST(
        dev_counter_,
        dev_pillar_count_,
        dev_x_coors_cpu,
        dev_y_coors_cpu,
        dev_num_points_per_pillar_cpu,
        dev_sparse_pillar_map_cpu,
        dev_pillar_count_histo_,
        max_num_pillars_,
        max_num_points_per_pillar_,
        grid_x_size_,
        num_inds_for_scan_
    );

    if(!ret)
    {
        goto LABEL_RET_FALSE_MakePillarIndexST;
    }

    ret = MakePillarFeatureST(
        dev_pillar_point_feature_cpu,
        dev_pillar_coors_cpu,
        dev_pillar_point_feature_in_coors_,
        dev_x_coors_cpu,
        dev_y_coors_cpu,
        dev_num_points_per_pillar_cpu,
        max_num_points_per_pillar_,
        num_point_feature_,
        grid_x_size_
    );

    dev_pillar_point_feature_crc = gfCalcFloatsCRC(dev_pillar_point_feature_cpu, max_num_pillars_*max_num_points_per_pillar_*num_point_feature_, 6);
    dev_pillar_coors_crc = gfCalcFloatsCRC(dev_pillar_coors_cpu, max_num_pillars_*4, 6);
    LOGPF("dev_pillar_point_feature_crc: 0x%x, dev_pillar_coors_crc: 0x%x", dev_pillar_point_feature_crc, dev_pillar_coors_crc);

LABEL_RET_FALSE_MakePillarIndexST:
    delete[] dev_x_coors_cpu;
    delete[] dev_y_coors_cpu;
    delete[] dev_num_points_per_pillar_cpu;
    delete[] dev_sparse_pillar_map_cpu;
    delete[] dev_pillar_point_feature_cpu;
    delete[] dev_pillar_coors_cpu;
    return ret;
}


bool PointpillarsOpsPreProc::TestCalcPillarMeanST(
    int32_t& dev_points_mean_crc,
    const float* points_ptr, 
    const int num_points
)
{
    bool ret = MakePillarHistoST(
        dev_pillar_point_feature_in_coors_,
        dev_pillar_count_histo_,
        points_ptr,
        num_points,
        max_num_points_per_pillar_, 
        grid_x_size_, 
        grid_y_size_,
        grid_z_size_, 
        min_x_range_, 
        min_y_range_, 
        min_z_range_, 
        pillar_x_size_,
        pillar_y_size_, 
        pillar_z_size_, 
        num_point_feature_
    );

    if(!ret)
    {
        return ret;
    }

    int* dev_x_coors_cpu = new int[max_num_pillars_];
    int* dev_y_coors_cpu = new int[max_num_pillars_];
    float* dev_num_points_per_pillar_cpu = new float[max_num_pillars_];
    int* dev_sparse_pillar_map_cpu = new int[num_inds_for_scan_*num_inds_for_scan_];
    memset(dev_x_coors_cpu, 0, max_num_pillars_*sizeof(int));
    memset(dev_y_coors_cpu, 0, max_num_pillars_*sizeof(int));
    memset(dev_num_points_per_pillar_cpu, 0, max_num_pillars_*sizeof(float));
    memset(dev_sparse_pillar_map_cpu, 0, num_inds_for_scan_*num_inds_for_scan_*sizeof(int));

    float* dev_pillar_point_feature_cpu = new float[max_num_pillars_*max_num_points_per_pillar_*num_point_feature_];
    float* dev_pillar_coors_cpu = new float[max_num_pillars_*4];
    memset(dev_pillar_point_feature_cpu, 0, max_num_pillars_*max_num_points_per_pillar_*num_point_feature_*sizeof(float));
    memset(dev_pillar_coors_cpu, 0, max_num_pillars_*4*sizeof(float));

    ret = MakePillarIndexST(
        dev_counter_,
        dev_pillar_count_,
        dev_x_coors_cpu,
        dev_y_coors_cpu,
        dev_num_points_per_pillar_cpu,
        dev_sparse_pillar_map_cpu,
        dev_pillar_count_histo_,
        max_num_pillars_,
        max_num_points_per_pillar_,
        grid_x_size_,
        num_inds_for_scan_
    );

    if(!ret)
    {
        goto LABEL_RET_FALSE_CalcPillarMeanST;
    }

    ret = MakePillarFeatureST(
        dev_pillar_point_feature_cpu,
        dev_pillar_coors_cpu,
        dev_pillar_point_feature_in_coors_,
        dev_x_coors_cpu,
        dev_y_coors_cpu,
        dev_num_points_per_pillar_cpu,
        max_num_points_per_pillar_,
        num_point_feature_,
        grid_x_size_
    );

    if(!ret)
    {
        goto LABEL_RET_FALSE_CalcPillarMeanST;
    }

    ret = CalcPillarMeanST(
        dev_points_mean_,
        num_point_feature_, 
        dev_pillar_point_feature_cpu, 
        dev_num_points_per_pillar_cpu, 
        max_num_pillars_,
        max_num_points_per_pillar_
    );

    dev_points_mean_crc = gfCalcFloatsCRC(dev_points_mean_, max_num_pillars_*3, 6);
    LOGPF("dev_points_mean_crc: 0x%x", dev_points_mean_crc);

LABEL_RET_FALSE_CalcPillarMeanST:
    delete[] dev_x_coors_cpu;
    delete[] dev_y_coors_cpu;
    delete[] dev_num_points_per_pillar_cpu;
    delete[] dev_sparse_pillar_map_cpu;
    delete[] dev_pillar_point_feature_cpu;
    delete[] dev_pillar_coors_cpu;
    return ret;
}