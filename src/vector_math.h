#ifndef VECTOR_MATH_H_
#define VECTOR_MATH_H_

#include <cmath>
#include <ostream>

struct Vec
{
    double x, y;
    Vec(double _x = 0, double _y = 0) : x(_x), y(_y) {}

    // vector + vector
    Vec operator+(const Vec &o) const { return Vec(x + o.x, y + o.y); }

    // vector - vector
    Vec operator-(const Vec &o) const { return Vec(x - o.x, y - o.y); }

    // vector * scalar  (Vec * s)
    Vec operator*(double s) const { return Vec(x * s, y * s); }

    // vector / scalar
    Vec operator/(double s) const { return Vec(x / s, y / s); }

    // +=
    Vec &operator+=(const Vec &o)
    {
        x += o.x;
        y += o.y;
        return *this;
    }

    // *=
    Vec &operator*=(double s)
    {
        x *= s;
        y *= s;
        return *this;
    }
};

// scalar * vector  (s * Vec)
inline Vec operator*(double s, const Vec &v)
{
    return Vec(v.x * s, v.y * s);
}

// dot product
inline double dot(const Vec &a, const Vec &b)
{
    return a.x * b.x + a.y * b.y;
}

// norm / magnitude
inline double norm(const Vec &v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

// printing
inline std::ostream &operator<<(std::ostream &os, const Vec &v)
{
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

#endif // VECTOR_MATH_H_
