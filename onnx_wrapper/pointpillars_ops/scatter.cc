#include "crc_checker.h"
#include "scatter.h"

PointpillarsOpsScatter::PointpillarsOpsScatter(
    const int num_threads,
    const int grid_x_size,
    const int grid_y_size
)
:   num_threads_(num_threads),
    grid_x_size_(grid_x_size),
    grid_y_size_(grid_y_size)
{
    RLOGI("PointpillarsOpsScatter num_threads_: %d, grid_x_size_: %d, grid_y_size_", num_threads_, grid_x_size_, grid_y_size_);
}

PointpillarsOpsScatter::~PointpillarsOpsScatter()
{}

bool PointpillarsOpsScatter::DoScatterST(
    float* const scattered_feature,
    const int pillar_count,
    const int* x_coors,
    const int* y_coors,
    const float* pfe_output
)
{
    RLOGI("DoScatterST pillar_count: %d", pillar_count);
    for(int i_pillar=0; i_pillar<pillar_count; i_pillar++)
    {
        const int x_ind = x_coors[i_pillar];
        const int y_ind = y_coors[i_pillar];
        for(int i_feature=0; i_feature<num_threads_; i_feature++)
        {
            const float feature = pfe_output[i_pillar * num_threads_ + i_feature];
            scattered_feature[i_feature * grid_y_size_ * grid_x_size_ + y_ind * grid_x_size_ + x_ind] = feature;
        }
    }
    return true;
}