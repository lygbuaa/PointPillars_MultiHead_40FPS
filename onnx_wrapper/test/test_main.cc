#include <gtest/gtest.h>
#include "logging_utils.h"

GTEST_API_ int main(int argc, char **argv)
{
    for(int i=0; i<argc; i++)
    {
        LOGPF("argv[%d] = %s\n", i, argv[i]);
    }
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
