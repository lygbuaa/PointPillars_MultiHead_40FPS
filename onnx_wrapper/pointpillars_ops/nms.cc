#include "nms.h"

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;
struct Point
{
    float x;
    float y;

    Point() {}
    Point(double _x, double _y)
    {
        x = _x, y = _y;
    }

    void set(float _x, float _y)
    {
        x = _x; y = _y;
    }

    Point operator +(const Point &b)const
    {
        return Point(x + b.x, y + b.y);
    }

    Point operator -(const Point &b)const
    {
        return Point(x - b.x, y - b.y);
    }
};

inline float cross(const Point &a, const Point &b)
{
    return a.x * b.y - a.y * b.x;
}

inline float cross(const Point &p1, const Point &p2, const Point &p0)
{
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

int check_rect_cross(const Point &p1, const Point &p2, const Point &q1, const Point &q2)
{
    int ret = std::min(p1.x,p2.x) <= std::max(q1.x,q2.x)  &&
              std::min(q1.x,q2.x) <= std::max(p1.x,p2.x) &&
              std::min(p1.y,p2.y) <= std::max(q1.y,q2.y) &&
              std::min(q1.y,q2.y) <= std::max(p1.y,p2.y);
    return ret;
}

inline int check_in_box2d(const float *box, const Point &p)
{
    //params: (7) [x, y, z, dx, dy, dz, heading]
    const float MARGIN = 1e-2;
    float center_x = box[0], center_y = box[1];
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);  // rotate the point in the opposite direction of box
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box[3] / 2 + MARGIN && fabs(rot_y) < box[4] / 2 + MARGIN);
}

inline int intersection(const Point &p1, const Point &p0, const Point &q1, const Point &q0, Point &ans)
{
    // fast exclusion
    if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

    // check cross standing
    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

    // calculate intersection of two lines
    float s5 = cross(q1, p1, p0);
    if(fabs(s5 - s1) > EPS){
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    }
    else
    {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return 1;
}

inline void rotate_around_center(const Point &center, const float angle_cos, const float angle_sin, Point &p)
{
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p.set(new_x, new_y);
}

inline int point_cmp(const Point &a, const Point &b, const Point &center)
{
    return atan2(a.y - center.y, a.x - center.x) > atan2(b.y - center.y, b.x - center.x);
}

inline float box_overlap(const float *box_a, const float *box_b)
{
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]

    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2, a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    Point center_a(box_a[0], box_a[1]);
    Point center_b(box_b[0], box_b[1]);

#ifdef DEBUG
    printf("a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", a_x1, a_y1, a_x2, a_y2, a_angle,
           b_x1, b_y1, b_x2, b_y2, b_angle);
    printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y, center_b.x, center_b.y);
#endif

    Point box_a_corners[5];
    box_a_corners[0].set(a_x1, a_y1);
    box_a_corners[1].set(a_x2, a_y1);
    box_a_corners[2].set(a_x2, a_y2);
    box_a_corners[3].set(a_x1, a_y2);

    Point box_b_corners[5];
    box_b_corners[0].set(b_x1, b_y1);
    box_b_corners[1].set(b_x2, b_y1);
    box_b_corners[2].set(b_x2, b_y2);
    box_b_corners[3].set(b_x1, b_y2);

    // get oriented corners
    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++){
#ifdef DEBUG
        printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
        printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    // get intersection of lines
    Point cross_points[16];
    Point poly_center;
    int cnt = 0, flag = 0;

    poly_center.set(0, 0);
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < 4; j++){
            flag = intersection(box_a_corners[i + 1], box_a_corners[i], box_b_corners[j + 1], box_b_corners[j], cross_points[cnt]);
            if (flag){
                poly_center = poly_center + cross_points[cnt];
                cnt++;
#ifdef DEBUG
                printf("Cross points (%.3f, %.3f): a(%.3f, %.3f)->(%.3f, %.3f), b(%.3f, %.3f)->(%.3f, %.3f) \n",
                    cross_points[cnt - 1].x, cross_points[cnt - 1].y,
                    box_a_corners[i].x, box_a_corners[i].y, box_a_corners[i + 1].x, box_a_corners[i + 1].y,
                    box_b_corners[i].x, box_b_corners[i].y, box_b_corners[i + 1].x, box_b_corners[i + 1].y);
#endif
            }
        }
    }

    // check corners
    for (int k = 0; k < 4; k++){
        if (check_in_box2d(box_a, box_b_corners[k])){
            poly_center = poly_center + box_b_corners[k];
            cross_points[cnt] = box_b_corners[k];
            cnt++;
#ifdef DEBUG
                printf("b corners in a: corner_b(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
        if (check_in_box2d(box_b, box_a_corners[k])){
            poly_center = poly_center + box_a_corners[k];
            cross_points[cnt] = box_a_corners[k];
            cnt++;
#ifdef DEBUG
                printf("a corners in b: corner_a(%.3f, %.3f)", cross_points[cnt - 1].x, cross_points[cnt - 1].y);
#endif
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // sort the points of polygon
    Point temp;
    for (int j = 0; j < cnt - 1; j++)
    {
        for (int i = 0; i < cnt - j - 1; i++)
        {
            if (point_cmp(cross_points[i], cross_points[i + 1], poly_center))
            {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

#ifdef DEBUG
    printf("cnt=%d\n", cnt);
    for (int i = 0; i < cnt; i++){
        printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x, cross_points[i].y);
    }
#endif

    // get the overlap areas
    float area = 0;
    for (int k = 0; k < cnt - 1; k++)
    {
        area += cross(cross_points[k] - cross_points[0], cross_points[k + 1] - cross_points[0]);
    }

    return fabs(area) / 2.0;
}

inline float iou_bev(const float *box_a, const float *box_b)
{
    // params box_a: [x, y, z, dx, dy, dz, heading]
    // params box_b: [x, y, z, dx, dy, dz, heading]
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float s_overlap = box_overlap(box_a, box_b);
    return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}

PointpillarsOpsNMS::PointpillarsOpsNMS(
    const int num_threads, 
    const int num_box_corners,
    const float nms_overlap_threshold
):
    num_threads_(num_threads),
    num_box_corners_(num_box_corners),
    nms_overlap_threshold_(nms_overlap_threshold)
{
}

PointpillarsOpsNMS::~PointpillarsOpsNMS()
{
}

bool PointpillarsOpsNMS::DoNmsST(
    int* const out_num_to_keep,
    long* const out_keep_inds,
    const int in_sorted_box_count,
    const float* in_sorted_box_for_nms
)
{
    /** dim=7 box: float[x, y, z, dx, dy, dz, heading] */
    int* in_sorted_box_mask = new int[in_sorted_box_count];
    /** mask>0: remove this box */
    memset(in_sorted_box_mask, 0, in_sorted_box_count * sizeof(int));
    *out_num_to_keep = 0;

    /** traverse to mark removed index */
    for(int i=0; i<in_sorted_box_count; i++)
    {
        const float *box_i = in_sorted_box_for_nms + i * 7;
        /** box_j already masked remove */
        if(in_sorted_box_mask[i] > 0)
        {
            continue;
        }
        for(int j=i+1; j<in_sorted_box_count; j++)
        {
            const float *box_j = in_sorted_box_for_nms + j * 7;
            /** box_j already masked remove */
            if(in_sorted_box_mask[j] > 0)
            {
                continue;
            }
            else
            {
                /** mark box_j to be removed */
                if (iou_bev(box_i, box_j) > nms_overlap_threshold_)
                {
                    in_sorted_box_mask[j] = 1;
                }
            }
        }
    }

    /** traverse to pick survivors */
    for(int i=0; i<in_sorted_box_count; i++)
    {
        if(in_sorted_box_mask[i] < 1)
        {
            out_keep_inds[*out_num_to_keep] = i;
            *out_num_to_keep += 1;
        }
    }

    return true;
}
