#ifndef UNIVERSAL_CONSTANTS_H_
#define UNIVERSAL_CONSTANTS_H_

#include <math.h>

namespace UniversalConstants
{
    constexpr double G = 6.67430e-11; // Gravitational constant, units: m^3 kg^-1 s^-2
    constexpr double G_scaled = 10.0; // Scaled gravitational constant for simulation
    constexpr double EPS0 = 8.854187817e-12; // Vacuum permittivity, units: F/m (farads per meter)
    constexpr double COULOMB_CONSTANT = 1.0 / (4.0 * M_PI * EPS0); // Coulomb's constant, units: N m^2 C^-2
    // physical constants for H+ (proton)
    constexpr double PROTON_MASS = 1.67262192369e-27;     // kg
    constexpr double ELEMENTARY_CHARGE = 1.602176634e-19; // C
} // namespace UniversalConstants

#endif // UNIVERSAL_CONSTANTS_H_