#include "time_utility.h"

static double compute_duration_time_windows(
    const struct STime &start, const struct STime &end)
{
#ifdef _WIN32
    LARGE_INTEGER Frequency;
    QueryPerformanceFrequency(&Frequency);
    return (end.m_t.QuadPart - start.m_t.QuadPart) * 1.0 / Frequency.QuadPart * 1000000;
#else
    return 0;
#endif
}

static double compute_duration_time_linux(
    const struct STime &start, const struct STime &end)
{
#ifdef __linux__
    return (end.m_t.tv_sec * 1000000 + end.m_t.tv_usec) * 1.0 -
           (start.m_t.tv_sec * 1000000 + start.m_t.tv_usec);
#else
    return 0;
#endif
}

double compute_duration_time(
    const struct STime &start, const struct STime &end)
{
#ifdef _WIN32
    return compute_duration_time_windows(start, end);
#else
    return compute_duration_time_linux(start, end);
#endif
}

void get_current_time(STime &t)
{
#ifdef _WIN32
    QueryPerformanceCounter(&(t.m_t));
#else
    gettimeofday(&(t.m_t), 0);
#endif
}

void sleep_for(int millisecond)
{
#ifdef _WIN32
    Sleep(millisecond);
#else
    usleep(1000 * millisecond);
#endif
}