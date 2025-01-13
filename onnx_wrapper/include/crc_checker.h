#ifndef __CRC_CHECKER_H__
#define __CRC_CHECKER_H__

#include <cmath>
#include <boost/crc.hpp>
#include "logging_utils.h"

static inline int32_t gfCalcBytesCRC(const void* u8_buffer, const size_t byte_count)
{
    boost::crc_32_type checker;
    // LOGPF("gfCalcBufferCRC byte_count: %ld", byte_count);
    checker.process_bytes(u8_buffer, byte_count);
    // LOGPF("gfCalcBufferCRC process_bytes done");
    int32_t result = checker.checksum();
    // LOGPF("gfCalcBufferCRC result: %x", result);
    return result;
}

static inline int32_t gfCalcFloatsCRC(const float* fp32_buffer, const size_t fp32_count, const int32_t precision=6)
{
    int32_t* s32_buffer = new int32_t[fp32_count];
    memset(s32_buffer, 0, fp32_count * sizeof(int32_t));
    for(size_t i=0; i<fp32_count; i++)
    {
        s32_buffer[i] = (int32_t)(fp32_buffer[i] * pow(10, precision));
    }
    int32_t result = gfCalcBytesCRC(s32_buffer, fp32_count * sizeof(int32_t));

    delete[] s32_buffer;
    return result;
}

static inline int32_t gfCalcFloatsSUM(const float* fp32_buffer, const size_t fp32_count, const int32_t precision=6)
{
    float sum = 0;
    for(size_t i=0; i<fp32_count; i++)
    {
        sum += (fp32_buffer[i] * pow(10, precision));
    }
    return (int32_t)sum;
}

#endif