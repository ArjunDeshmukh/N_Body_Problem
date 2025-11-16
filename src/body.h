#ifndef BODY_H_
#define BODY_H_

#include "vector_math.h"
#include <SFML/Graphics.hpp>

class Body
{
public:
    Body(double mass, double q, Vec pos, Vec vel, double radius, sf::Color color, bool alive)
        : mass(mass), q(q), pos(pos), vel(vel), radius(radius), color(color), alive(alive) {}
    Body(double mass, Vec pos, Vec vel, double radius, sf::Color color, bool alive)
        : mass(mass), q(0.0), pos(pos), vel(vel), radius(radius), color(color), alive(alive) {}   
    Body()
     : mass(0.0), q(0.0), pos(Vec(0, 0)), vel(Vec(0, 0)), radius(1.0), color(sf::Color::White), alive(false) {} 
    
    ~Body() = default;
    
    double mass = 0.0;
    double q = 0.0; // charge
    Vec pos = Vec(0, 0);
    Vec vel = Vec(0, 0);
    double radius = 1.0;
    sf::Color color = sf::Color::White;
    bool alive = false;
};

#endif // BODY_H_