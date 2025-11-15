#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "create_window.h"


#define G 6.67430e-1

struct Vec
{
    double x, y;
    Vec(double _x = 0, double _y = 0) : x(_x), y(_y) {}
    Vec operator+(const Vec &o) const { return Vec(x + o.x, y + o.y); }
    Vec operator-(const Vec &o) const { return Vec(x - o.x, y - o.y); }
    Vec operator*(double s) const { return Vec(x * s, y * s); }
    Vec operator/(double s) const { return Vec(x / s, y / s); }
    Vec &operator+=(const Vec &o)
    {
        x += o.x;
        y += o.y;
        return *this;
    }
};

double norm(const Vec &v) { return std::sqrt(v.x * v.x + v.y * v.y); }

struct Body
{
    double mass;
    Vec pos;
    Vec vel;
    double radius;
    sf::Color color;
    bool alive = true;
};

int main()
{
    const double dt = 0.01;
    // simulation speed: how many fixed `dt` steps to run per frame (can be fractional)
    double simSpeed = 20.0; // default: 20 dt-steps per frame (faster-than-real-time)
    double simStepsAcc = 0.0; // accumulator for fractional steps
    const int winW = 1000;
    const int winH = 800;

    // Create window
    sf::RenderWindow window;
    CreateWindow(window, winW, winH,  "Three-Body Gravity: Perfectly Inelastic Collisions");

    // --- Initial conditions ---
    std::vector<Body> bodies(3);
    std::vector<Body> init_bodies(3);

    // --- Figure-eight (choreography) initial conditions ---
    // Based on the canonical figure-eight initial values (Chenciner–Montgomery),
    // scaled & translated for a 1000x800 window. Equal masses.

    const double centerX = 500.0;
    const double centerY = 400.0;
    const double L_scale = 80.0;     // spatial scale (pixels per unit)
    const double mass_each = 1000.0; // increase mass so visible speeds are reasonable

    // canonical (unit) figure-eight coords & velocities (unit masses, G=1)
    const Vec r1_unit(0.97000436, -0.24308753);
    const Vec r2_unit(-0.97000436, 0.24308753);
    const Vec r3_unit(0.0, 0.0);

    const Vec v1_unit(0.4662036850, 0.4323657300);
    const Vec v2_unit(0.4662036850, 0.4323657300);
    const Vec v3_unit(-0.93240737, -0.86473146);

    // scale factor for velocities:
    // v_new = v_unit * sqrt( G * mass / L_scale )
    // (we keep your G = 6.67430e-1 from the code)
    const double vel_scale = std::sqrt(6.67430e-1 * mass_each / L_scale);

    // final screen positions and velocities
    init_bodies[0] = {
        mass_each,
        {(float)(centerX + r1_unit.x * L_scale), (float)(centerY + r1_unit.y * L_scale)},
        {(double)(v1_unit.x * vel_scale), (double)(v1_unit.y * vel_scale)},
        10.0, // radius (for drawing)
        sf::Color::Red,
        true};
    init_bodies[1] = {
        mass_each,
        {(float)(centerX + r2_unit.x * L_scale), (float)(centerY + r2_unit.y * L_scale)},
        {(double)(v2_unit.x * vel_scale), (double)(v2_unit.y * vel_scale)},
        10.0,
        sf::Color::Blue,
        true};
    init_bodies[2] = {
        mass_each,
        {(float)(centerX + r3_unit.x * L_scale), (float)(centerY + r3_unit.y * L_scale)},
        {(double)(v3_unit.x * vel_scale), (double)(v3_unit.y * vel_scale)},
        10.0,
        sf::Color::Green,
        true};

    // Above configuration is a stable 3-body orbit, bodies never collide.
    // Let's nudge one body slightly to induce collisions.
    const double vel_error_injection = 0.1*vel_scale;
    init_bodies[2].vel.x += vel_error_injection;

    bodies = init_bodies;

    std::vector<sf::VertexArray> trails(3, sf::VertexArray(sf::LinesStrip));

    bool running = true;
    bool showVectors = true;
    double simTime = 0.0;
    sf::View view = window.getDefaultView();

    while (window.isOpen())
    {
        sf::Event ev;
        while (window.pollEvent(ev))
        {
            if (ev.type == sf::Event::Closed)
                window.close();
            if (ev.type == sf::Event::KeyPressed)
            {
                if (ev.key.code == sf::Keyboard::Space)
                    running = !running;
                if (ev.key.code == sf::Keyboard::Up)
                    simSpeed *= 2.0;
                if (ev.key.code == sf::Keyboard::Down)
                    simSpeed = std::max(0.01, simSpeed / 2.0);
                if (ev.key.code == sf::Keyboard::R)
                {
                    // reset
                    bodies = init_bodies;
                    for (auto &t : trails)
                        t.clear();
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::V)
                    showVectors = !showVectors;
            }
        }

        if (running)
        {
            // accumulate number of dt steps to run this frame; simSpeed is steps/frame
            simStepsAcc += simSpeed;
            while (simStepsAcc >= 1.0)
            {
                // compute accelerations based on current positions
                std::vector<Vec> acc(bodies.size(), Vec(0, 0));

                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    if (!bodies[i].alive)
                        continue;
                    for (size_t j = i + 1; j < bodies.size(); ++j)
                    {
                        if (!bodies[j].alive)
                            continue;

                        Vec r = bodies[j].pos - bodies[i].pos;
                        double d = norm(r);
                        if (d < 1e-6)
                            continue;
                        double minDist = bodies[i].radius + bodies[j].radius;

                        // collision
                        if (d <= minDist)
                        {
                            double M = bodies[i].mass + bodies[j].mass;
                            Vec v_new = (bodies[i].vel * bodies[i].mass + bodies[j].vel * bodies[j].mass) / M;
                            Vec p_new = (bodies[i].pos * bodies[i].mass + bodies[j].pos * bodies[j].mass) / M;

                            bodies[i].mass = M;
                            bodies[i].pos = p_new;
                            bodies[i].vel = v_new;
                            bodies[i].radius = std::sqrt(bodies[i].radius * bodies[i].radius + bodies[j].radius * bodies[j].radius) * 1.08;

                            bodies[j].alive = false;
                            bodies[j].radius = 0.0;
                            bodies[j].mass = 0.0;
                            trails[j].clear();
                            continue;
                        }

                        Vec dir = r / d;
                        double a_i = G * bodies[j].mass / (d * d);
                        double a_j = G * bodies[i].mass / (d * d);

                        acc[i] += dir * a_i;
                        acc[j] += dir * (-a_j);
                    }
                }

                // integrate motion for exactly one dt
                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    if (!bodies[i].alive)
                        continue;
                    bodies[i].vel += acc[i] * dt;
                    bodies[i].pos += bodies[i].vel * dt;

                    sf::Vertex v(sf::Vector2f((float)bodies[i].pos.x, (float)bodies[i].pos.y));
                    trails[i].append(v);
                    if (trails[i].getVertexCount() > 500)
                    {
                        sf::VertexArray temp(sf::LinesStrip);
                        for (size_t k = 1; k < trails[i].getVertexCount(); ++k)
                            temp.append(trails[i][k]);
                        trails[i] = temp;
                    }
                }

                simTime += dt;
                simStepsAcc -= 1.0;
            }
        }

        // Calculate center of mass and update camera
        Vec centerOfMass(0, 0);
        double totalMass = 0;
        for (const auto &b : bodies)
        {
            if (b.alive)
            {
                centerOfMass += b.pos * b.mass;
                totalMass += b.mass;
            }
        }
        if (totalMass > 0)
        {
            centerOfMass = centerOfMass / totalMass;
        }
        view.setCenter((float)centerOfMass.x, (float)centerOfMass.y);
        window.setView(view);

        // draw
        window.clear(sf::Color::Black);

        for (auto &t : trails)
            if (t.getVertexCount() > 1)
                window.draw(t);

        for (const auto &b : bodies)
        {
            if (!b.alive)
                continue;
            sf::CircleShape c((float)b.radius);
            c.setOrigin((float)b.radius, (float)b.radius);
            c.setPosition((float)b.pos.x, (float)b.pos.y);
            c.setFillColor(b.color);
            window.draw(c);

            if (showVectors)
            {
                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f((float)b.pos.x, (float)b.pos.y)),
                    sf::Vertex(sf::Vector2f((float)(b.pos.x + b.vel.x), (float)(b.pos.y + b.vel.y)))};
                window.draw(line, 2, sf::Lines);
            }
        }

        // Reset view for HUD text (screen coordinates)
        window.setView(window.getDefaultView());

        sf::Font font;
        if (font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        {
            sf::Text txt;
            txt.setFont(font);
            txt.setCharacterSize(14);
            txt.setFillColor(sf::Color::White);
            std::string info = "Space: pause/play  |  R: reset  |  V: toggle vectors\n";
            info += "Sim time: " + std::to_string(simTime).substr(0, 6);
            info += "  |  Speed: " + std::to_string(simSpeed).substr(0, 6);
            txt.setString(info);
            txt.setPosition(8, 8);
            window.draw(txt);
        }

        window.display();
    }

    return 0;
}
