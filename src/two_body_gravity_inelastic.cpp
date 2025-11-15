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
    double simSpeed = 20.0;
    double simStepsAcc = 0.0;
    const int winW = 1000;
    const int winH = 800;

    // Create window
    sf::RenderWindow window;
    CreateWindow(window, winW, winH,  "Three-Body Gravity: Perfectly Inelastic Collisions");

    // two bodies
    std::vector<Body> bodies(2);
    std::vector<Body> init_bodies(2);

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

    bodies = init_bodies;

    std::vector<sf::VertexArray> trails(2, sf::VertexArray(sf::LinesStrip));

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
            simStepsAcc += simSpeed;
            while (simStepsAcc >= 1.0)
            {
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

                        // collision -> perfectly inelastic merge
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

                // integrate exactly one dt
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

        // center camera on center of mass
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
            centerOfMass = centerOfMass / totalMass;

        view.setCenter((float)centerOfMass.x, (float)centerOfMass.y);
        window.setView(view);

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

        // HUD in screen coordinates
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
