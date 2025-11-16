#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include "create_window.h"
#include <random>
#include <iostream>
#include "body.h"
#include "vector_math.h"

static void init_system(std::vector<Body> &init_bodies, int N, int winW, int winH)
{
    init_bodies.clear();
    init_bodies.resize(N);

    const double centerX = winW * 0.5;
    const double centerY = winH * 0.5;

    const sf::Color palette[6] = {sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow, sf::Color::Magenta, sf::Color::Cyan};

    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<float> radius(0.1, 0.1);
    static std::uniform_real_distribution<float> mass(5.0, 10.0);
    static std::uniform_real_distribution<float> pos_x(-centerX, centerX);
    static std::uniform_real_distribution<float> pos_y(-centerY, centerY);
    static std::uniform_real_distribution<float> vel_x(-10.0, 10.0);
    static std::uniform_real_distribution<float> vel_y(-10.0, 10.0);

    for (int i = 0; i < N; ++i)
    {
        init_bodies[i].mass = mass(rng);
        init_bodies[i].pos = Vec(centerX + pos_x(rng), centerY + pos_y(rng));
        init_bodies[i].vel = Vec(vel_x(rng), vel_y(rng));
        init_bodies[i].radius = radius(rng);
        //init_bodies[i].color = palette[i % 6];
        init_bodies[i].color = sf::Color::Red;
        init_bodies[i].alive = true;
    }

    if (N == 1)
    {
        init_bodies[0].vel.x = 10.0;
    }
}

int main()
{
    const double dt = 0.01;
    double simSpeed = 20.0;
    double simStepsAcc = 0.0;
    const int winW = 1000;
    const int winH = 800;
    const double e_obj = 0.9;
    const double e_wall = 0.8;

    double vel_vector_draw_scale = 10.0;

    // Create window
    sf::RenderWindow window;
    CreateWindow(window, winW, winH, "N-Body Collisions");

    // World box half-sizes (in pixels). The visual box remains fixed on screen,
    // but wall collisions are applied to world positions mapped to the current view center.
    const double boxHalfW = 400.0;
    const double boxHalfH = 300.0;

    // Create a screen-space drawable rectangle (fixed on screen)
    sf::RectangleShape boxShape(sf::Vector2f((float)(boxHalfW * 2.0), (float)(boxHalfH * 2.0)));
    boxShape.setPosition((float)((winW * 0.5) - boxHalfW), (float)((winH * 0.5) - boxHalfH));
    boxShape.setFillColor(sf::Color::Transparent);
    boxShape.setOutlineColor(sf::Color(200, 200, 200));
    boxShape.setOutlineThickness(2.0f);

    int N = 10000;
    std::vector<Body> bodies;
    std::vector<Body> init_bodies;
    init_system(init_bodies, N, boxHalfW * 2.0, boxHalfH * 2.0);
    bodies = init_bodies;
    std::vector<sf::VertexArray> trails(N, sf::VertexArray(sf::LinesStrip));

    bool running = true;
    bool showVectors = false;
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
                    N = std::min(1000, N + 1);
                    init_system(init_bodies, N, boxHalfW * 2.0, boxHalfH * 2.0);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::M)
                {
                    N = std::max(2, N - 1);
                    init_system(init_bodies, N, boxHalfW * 2.0, boxHalfH * 2.0);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::R)
                {
                    bodies = init_bodies;
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
                // compute current world-space box bounds from the view center
                const sf::Vector2f viewCenter = view.getCenter();
                const double worldBoxMinX = (double)viewCenter.x - boxHalfW;
                const double worldBoxMaxX = (double)viewCenter.x + boxHalfW;
                const double worldBoxMinY = (double)viewCenter.y - boxHalfH;
                const double worldBoxMaxY = (double)viewCenter.y + boxHalfH;

                std::vector<Vec> acc(bodies.size(), Vec(0, 0));

                // integrate
                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    bodies[i].pos += bodies[i].vel * dt;
                    bodies[i].vel += acc[i] * dt;

                    if (trails[i].getVertexCount() > 500)
                    {
                        sf::VertexArray temp(sf::LinesStrip);
                        for (size_t k = 1; k < trails[i].getVertexCount(); ++k)
                            temp.append(trails[i][k]);
                        trails[i] = temp;
                    }
                }

                std::vector<Vec> impulse_vec(bodies.size(), Vec(0, 0));

                // handle partially elastic collisions with the axis-aligned box and objects with each other
                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    if (!bodies[i].alive)
                        continue;

                    // wall collisions: fully elastic (reverse perpendicular velocity)
                    if ((bodies[i].pos.x - bodies[i].radius) < worldBoxMinX)
                    {
                        Vec temp_impulse_vec = Vec(-(1 + e_wall) * bodies[i].vel.x, 0.0);
                        impulse_vec[i] += temp_impulse_vec;
                        bodies[i].pos.x = worldBoxMinX + bodies[i].radius;
                        // bodies[i].vel.x = -bodies[i].vel.x;
                    }
                    else if ((bodies[i].pos.x + bodies[i].radius) > worldBoxMaxX)
                    {
                        Vec temp_impulse_vec = Vec(-(1 + e_wall) * bodies[i].vel.x, 0.0);
                        impulse_vec[i] += temp_impulse_vec;
                        bodies[i].pos.x = worldBoxMaxX - bodies[i].radius;
                        // bodies[i].vel.x = -bodies[i].vel.x;
                    }

                    if ((bodies[i].pos.y - bodies[i].radius) < worldBoxMinY)
                    {
                        Vec temp_impulse_vec = Vec(0.0, -(1 + e_wall) * bodies[i].vel.y);
                        impulse_vec[i] += temp_impulse_vec;
                        bodies[i].pos.y = worldBoxMinY + bodies[i].radius;
                        // bodies[i].vel.y = -bodies[i].vel.y;
                    }
                    else if ((bodies[i].pos.y + bodies[i].radius) > worldBoxMaxY)
                    {
                        Vec temp_impulse_vec = Vec(0.0, -(1 + e_wall) * bodies[i].vel.y);
                        impulse_vec[i] += temp_impulse_vec;
                        bodies[i].pos.y = worldBoxMaxY - bodies[i].radius;
                        // bodies[i].vel.y = -bodies[i].vel.y;
                    }

                    // Collisions of objects with each other
                    for (size_t j = i + 1; j < bodies.size(); ++j)
                    {
                        if (!bodies[j].alive)
                            continue;

                        Vec r = bodies[j].pos - bodies[i].pos;
                        double d = norm(r);
                        double minDist = bodies[i].radius + bodies[j].radius;

                        if (d < minDist)
                        {
                            Vec vel_rel = bodies[i].vel - bodies[j].vel;
                            Vec norm_r = r/d;

                            double vel_rel_mag_along_normal = dot(vel_rel, norm_r);

                            double mj = bodies[j].mass;
                            double mi = bodies[i].mass;

                            double impulse_scalar = -(1 + e_obj) * vel_rel_mag_along_normal / (1 / mj + 1 / mi);

                            impulse_vec[i] += norm_r * (impulse_scalar / mi);
                            impulse_vec[j] += norm_r * (-impulse_scalar / mj);

                            bodies[i].pos += norm_r * (-1.0 * (minDist - d));
                            bodies[j].pos += norm_r * ( 1.0 * (minDist - d));
                        }
                    }
                }

                // apply impulse to velocity vector
                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    bodies[i].vel += impulse_vec[i];
                }

                simTime += dt;
                simStepsAcc -= 1.0;
            }
        }

        // Calculate total energy
        double total_energy = 0.0;
        for (const auto &b : bodies)
        {
            total_energy += 0.5 * b.mass * norm(b.vel) * norm(b.vel);
        }

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
                sf::Vertex line[] = {sf::Vertex(sf::Vector2f((float)b.pos.x, (float)b.pos.y)), sf::Vertex(sf::Vector2f((float)(b.pos.x + vel_vector_draw_scale*b.vel.x), (float)(b.pos.y + vel_vector_draw_scale*b.vel.y)))};
                window.draw(line, 2, sf::Lines);
            }
        }

        window.setView(window.getDefaultView());
        // draw the screen-fixed box edges so they are visible on the HUD layer
        window.draw(boxShape);
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
            info += "  | Total Energy: " + std::to_string(total_energy/1.0e3).substr(0, 6) + " kJ";
            txt.setString(info);
            txt.setPosition(8, 8);
            window.draw(txt);
        }

        window.display();
    }

    return 0;
}
