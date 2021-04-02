#pragma once

#include <iostream>
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <unistd.h>
#endif

struct STime
{
#ifdef _WIN32
    LARGE_INTEGER m_t;
#else
    struct timeval m_t;
#endif
};

void get_current_time(STime &t);

double compute_duration_time(
    const struct STime &start, const struct STime &end);

void sleep_for(int millisecond);

#define evaluate_time(func, msg, num)                                             \
    {                                                                             \
        STime t1, t2;                                                             \
        get_current_time(t1);                                                     \
        for (int idx = 0; idx < num; idx++)                                       \
        {                                                                         \
            func;                                                                 \
        }                                                                         \
        get_current_time(t2);                                                     \
        std::cout << (msg) << compute_duration_time(t1, t2)/num/1000 << "ms" << std::endl; \
    }
