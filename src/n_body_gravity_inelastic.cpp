#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
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

// Initialize a system of N bodies. If N==3, optionally create the figure-eight choreography;
// otherwise generate a random distribution of masses/positions/velocities.
static void init_system(std::vector<Body> &init_bodies, int N, int winW, int winH)
{
    init_bodies.clear();
    init_bodies.resize(N);

    if (N == 2)
    {
        // central massive body (A)
        init_bodies[0].mass = 50000.0;
        init_bodies[0].pos = {(double)(winW * 0.5), (double)(winH * 0.5)};
        init_bodies[0].vel = {0.0, 0.0};
        init_bodies[0].radius = 22.0;
        init_bodies[0].color = sf::Color::Red;
        init_bodies[0].alive = true;

        // satellite (B)
        const double orbitR = 200.0;
        init_bodies[1].mass = 10.0;
        init_bodies[1].pos = {init_bodies[0].pos.x + orbitR, init_bodies[0].pos.y};
        double v_circ = std::sqrt(G * init_bodies[0].mass / orbitR);
        // perpendicular velocity (counter-clockwise)
        init_bodies[1].vel = {0.0, -v_circ};
        init_bodies[1].radius = 8.0;
        init_bodies[1].color = sf::Color::Blue;
        init_bodies[1].alive = true;

        // conserve momentum (give small recoil to central mass so total momentum = 0)
        init_bodies[0].vel.x = -(init_bodies[1].mass * init_bodies[1].vel.x) / init_bodies[0].mass;
        init_bodies[0].vel.y = -(init_bodies[1].mass * init_bodies[1].vel.y) / init_bodies[0].mass;
        return;
    }
    else if (N == 3)
    {
        // canonical figure-eight scaled
        const double centerX = winW * 0.5;
        const double centerY = winH * 0.5;
        const double L_scale = 80.0;
        const double mass_each = 1000.0;
        const Vec r1_unit(0.97000436, -0.24308753);
        const Vec r2_unit(-0.97000436, 0.24308753);
        const Vec r3_unit(0.0, 0.0);
        const Vec v1_unit(0.4662036850, 0.4323657300);
        const Vec v2_unit(0.4662036850, 0.4323657300);
        const Vec v3_unit(-0.93240737, -0.86473146);
        const double vel_scale = std::sqrt(6.67430e-1 * mass_each / L_scale);

        init_bodies[0] = {mass_each, {(float)(centerX + r1_unit.x * L_scale), (float)(centerY + r1_unit.y * L_scale)}, {(double)(v1_unit.x * vel_scale), (double)(v1_unit.y * vel_scale)}, 10.0, sf::Color::Red, true};
        init_bodies[1] = {mass_each, {(float)(centerX + r2_unit.x * L_scale), (float)(centerY + r2_unit.y * L_scale)}, {(double)(v2_unit.x * vel_scale), (double)(v2_unit.y * vel_scale)}, 10.0, sf::Color::Blue, true};
        init_bodies[2] = {mass_each, {(float)(centerX + r3_unit.x * L_scale), (float)(centerY + r3_unit.y * L_scale)}, {(double)(v3_unit.x * vel_scale), (double)(v3_unit.y * vel_scale)}, 10.0, sf::Color::Green, true};
        return;
    }

    // Place N bodies along a figure-eight (Gerono lemniscate) curve and
    // set velocities tangent to the curve so the system starts as a choreography-like setup.
    const double PI = std::acos(-1.0);
    const double centerX = winW * 0.5;
    const double centerY = winH * 0.5;
    const double L_scale = std::min(winW, winH) * 0.32; // spatial scale
    const double mass_each = 1000.0;
    const double vel_scale = std::sqrt(6.67430e-1 * mass_each / L_scale);

    const sf::Color palette[6] = {sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow, sf::Color::Magenta, sf::Color::Cyan};

    for (int i = 0; i < N; ++i)
    {
        double t = 2.0 * PI * double(i) / double(N);
        // Gerono lemniscate param: x = sin(t), y = 0.5*sin(2t)
        double xu = std::sin(t);
        double yu = 0.5 * std::sin(2.0 * t);
        // derivative (tangent) dx/dt = cos(t), dy/dt = cos(2t)
        double dx = std::cos(t);
        double dy = std::cos(2.0 * t);

        double m = mass_each;
        double radius = std::cbrt(m) * 0.6 + 3.0;
        init_bodies[i].mass = m;
        init_bodies[i].pos = Vec(centerX + xu * L_scale, centerY + yu * L_scale);
        init_bodies[i].vel = Vec(dx * vel_scale, dy * vel_scale);
        init_bodies[i].radius = radius;
        init_bodies[i].color = palette[i % 6];
        init_bodies[i].alive = true;
    }
}

int main()
{
    const double dt = 0.01;
    // simulation speed: how many fixed `dt` steps to run per frame (can be fractional)
    double simSpeed = 20.0;   // default: 20 dt-steps per frame (faster-than-real-time)
    double simStepsAcc = 0.0; // accumulator for fractional steps
    const int winW = 1000;
    const int winH = 800;

    // Create window
    sf::RenderWindow window;
    CreateWindow(window, winW, winH, "N-Body Gravity: Perfectly Inelastic Collisions");

    // --- Initial conditions (N-body) ---
    int N = 7; // default number of bodies
    std::vector<Body> bodies;
    std::vector<Body> init_bodies;
    init_system(init_bodies, N, winW, winH);
    bodies = init_bodies;
    std::vector<sf::VertexArray> trails(N, sf::VertexArray(sf::LinesStrip));

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
                if (ev.key.code == sf::Keyboard::N)
                {
                    // increase N and reinitialize
                    N = std::min(1000, N + 1);
                    init_system(init_bodies, N, winW, winH);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::M)
                {
                    // decrease N and reinitialize
                    N = std::max(2, N - 1);
                    init_system(init_bodies, N, winW, winH);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    simTime = 0.0;
                }
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
            info += "  |  N: " + std::to_string((int)bodies.size());
            txt.setString(info);
            txt.setPosition(8, 8);
            window.draw(txt);
        }

        window.display();
    }

    return 0;
}
