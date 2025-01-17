#ifndef __POINTPILLARS_OPS_SCATTER_H__
#define __POINTPILLARS_OPS_SCATTER_H__

class PointpillarsOpsScatter
{
public:
    PointpillarsOpsScatter(
        const int num_threads,
        const int grid_x_size,
        const int grid_y_size
    );
    ~PointpillarsOpsScatter();

    bool DoScatterST(
        float* const scattered_feature,
        const int pillar_count,
        const int* x_coors,
        const int* y_coors,
        const float* pfe_output
    );

private:
    /** num_threads_ should be the same with PFE model output channel, which is (64) */
    const int num_threads_;
    const int grid_x_size_;
    const int grid_y_size_;
};

#endif