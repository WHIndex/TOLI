#ifndef __SIPP_BASE_H__
#define __SIPP_BASE_H__

#include <limits>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// Linear regression model
template <class T>
class LinearModel_sipp
{
public:
    double a = 0; // slope
    long double b = 0; // intercept

    LinearModel_sipp() = default;
    LinearModel_sipp(double a, long double b) : a(a), b(b) {}
    explicit LinearModel_sipp(const LinearModel_sipp &other) : a(other.a), b(other.b) {}

    inline int predict(T key) const
    {
        return std::floor(a * static_cast<long double>(key) + b);
    }

    inline double predict_double(T key) const
    {
        return a * static_cast<long double>(key) + b;
    }
};

#endif // __SIPP_BASE_H__
