// osmosis.cpp
// Single-window fixed version.
// 20 px pores, 8 pores, pores closed at start, press O to toggle.

#include <SFML/Graphics.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include "body.h"
#include "vector_math.h"

// Helper: clamp
template<typename T>
T clamp(T v, T a, T b) { return (v < a) ? a : (v > b) ? b : v; }

int main()
{
    // --- Window / world ---
    const int winW = 1000;
    const int winH = 1000;
    sf::RenderWindow window(sf::VideoMode(winW, winH), "Osmosis (fixed)");
    window.setFramerateLimit(120);

    // world box centered in the window
    const double boxHalfW = 400.0;
    const double boxHalfH = 300.0;
    const double worldW = boxHalfW * 2.0;
    const double worldH = boxHalfH * 2.0;
    const double worldOriginX = (winW * 0.5) - boxHalfW;   // left
    const double worldOriginY = (winH * 0.5) - boxHalfH;   // top
    const double worldMinX = worldOriginX;
    const double worldMaxX = worldOriginX + worldW;
    const double worldMinY = worldOriginY;
    const double worldMaxY = worldOriginY + worldH;

    // visible box outline
    sf::RectangleShape boxShape(sf::Vector2f((float)worldW, (float)worldH));
    boxShape.setPosition((float)worldOriginX, (float)worldOriginY);
    boxShape.setFillColor(sf::Color::Transparent);
    boxShape.setOutlineThickness(2.0f);
    boxShape.setOutlineColor(sf::Color(90,90,90));

    // membrane properties
    const double membrane_x = worldOriginX + boxHalfW; // vertical center x
    const float membrane_thickness = 3.0f;

    // Pores: 8 pores, each vertical segment with height pore_h and horizontal opening pore_width
    const int num_pores = 8;
    const double pore_height = 25.0;
    const double pore_width = 20.0; // **your choice A: 20 px**
    std::vector<std::pair<double,double>> pores; // (y1,y2) in world coords (top,bottom)

    {
        double spacing = (worldH - 40.0) / (num_pores + 1);
        for (int i = 0; i < num_pores; ++i)
        {
            double py = worldOriginY + 20.0 + (i+1)*spacing;
            pores.emplace_back(py - pore_height*0.5, py + pore_height*0.5);
        }
    }

    // physics params
    const double dt = 0.01;
    double simSpeed = 20.0;
    double simAcc = 0.0;
    const double e_obj = 0.9;
    const double e_wall = 0.85;

    // particles
    const int N = 300;
    std::vector<Body> bodies(N);
    std::vector<Vec> prev_pos(N);
    std::vector<char> is_solute(N, 0);

    // RNG
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> radius_rng(2.0f, 4.5f);
    std::uniform_real_distribution<float> mass_rng(1.0f, 2.0f);
    std::uniform_real_distribution<float> vel_rng(-3.0f, 3.0f); // moderate speeds (B)
    std::uniform_real_distribution<float> pos_y_rng((float)worldMinY + 10.0f, (float)worldMaxY - 10.0f);

    // ensure left and right initial X ranges are safely away from membrane
    const double left_min_x  = worldMinX + 10.0;
    const double left_max_x  = membrane_x - 60.0;   // 60 px away to be safe
    const double right_min_x = membrane_x + 60.0;
    const double right_max_x = worldMaxX - 10.0;
    std::uniform_real_distribution<float> pos_x_left((float)left_min_x, (float)left_max_x);
    std::uniform_real_distribution<float> pos_x_right((float)right_min_x, (float)right_max_x);

    const double solute_fraction_on_right = 0.35;
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    // init bodies
    for (int i = 0; i < N; ++i)
    {
        bodies[i].mass = mass_rng(rng);
        bodies[i].radius = radius_rng(rng);
        bodies[i].alive = true;

        if (i < N/2)
        {
            // left: solvent only
            bodies[i].pos = Vec(pos_x_left(rng), pos_y_rng(rng));
            bodies[i].vel = Vec(vel_rng(rng), vel_rng(rng));
            bodies[i].color = sf::Color::Cyan;
            is_solute[i] = 0;
        }
        else
        {
            // right: mixture; solute only here
            bodies[i].pos = Vec(pos_x_right(rng), pos_y_rng(rng));
            bodies[i].vel = Vec(vel_rng(rng), vel_rng(rng));
            if (u01(rng) < solute_fraction_on_right)
            {
                is_solute[i] = 1;
                bodies[i].color = sf::Color::Yellow;
            }
            else
            {
                is_solute[i] = 0;
                bodies[i].color = sf::Color::Cyan;
            }
        }
    }

    // control flags
    bool running = true;
    bool pores_open = false; // start closed
    bool showVectors = false;
    double simTime = 0.0;

    // Main loop
    while (window.isOpen())
    {
        // events
        sf::Event ev;
        while (window.pollEvent(ev))
        {
            if (ev.type == sf::Event::Closed) window.close();
            if (ev.type == sf::Event::KeyPressed)
            {
                if (ev.key.code == sf::Keyboard::Space) running = !running;
                if (ev.key.code == sf::Keyboard::Up) simSpeed *= 2.0;
                if (ev.key.code == sf::Keyboard::Down) simSpeed = std::max(0.01, simSpeed/2.0);
                if (ev.key.code == sf::Keyboard::O) pores_open = !pores_open;
                if (ev.key.code == sf::Keyboard::V) showVectors = !showVectors;
                if (ev.key.code == sf::Keyboard::R)
                {
                    // reset particles, keep pores closed
                    for (int i = 0; i < N; ++i)
                    {
                        bodies[i].mass = mass_rng(rng);
                        bodies[i].radius = radius_rng(rng);
                        bodies[i].alive = true;
                        if (i < N/2)
                        {
                            bodies[i].pos = Vec(pos_x_left(rng), pos_y_rng(rng));
                            bodies[i].vel = Vec(vel_rng(rng), vel_rng(rng));
                            bodies[i].color = sf::Color::Cyan;
                            is_solute[i] = 0;
                        }
                        else
                        {
                            bodies[i].pos = Vec(pos_x_right(rng), pos_y_rng(rng));
                            bodies[i].vel = Vec(vel_rng(rng), vel_rng(rng));
                            if (u01(rng) < solute_fraction_on_right)
                            {
                                is_solute[i] = 1;
                                bodies[i].color = sf::Color::Yellow;
                            }
                            else
                            {
                                is_solute[i] = 0;
                                bodies[i].color = sf::Color::Cyan;
                            }
                        }
                    }
                    pores_open = false;
                    simTime = 0.0;
                }
            }
        }

        // physics updates
        if (running)
        {
            simAcc += simSpeed;
            while (simAcc >= 1.0)
            {
                // store previous positions
                for (int i = 0; i < N; ++i) prev_pos[i] = bodies[i].pos;

                // integrate positions
                for (int i = 0; i < N; ++i)
                    bodies[i].pos += bodies[i].vel * dt;

                // box walls (clamp and reflect)
                for (int i = 0; i < N; ++i)
                {
                    if ((bodies[i].pos.x - bodies[i].radius) < worldMinX)
                    {
                        bodies[i].pos.x = worldMinX + bodies[i].radius;
                        bodies[i].vel.x = -e_wall * bodies[i].vel.x;
                    }
                    if ((bodies[i].pos.x + bodies[i].radius) > worldMaxX)
                    {
                        bodies[i].pos.x = worldMaxX - bodies[i].radius;
                        bodies[i].vel.x = -e_wall * bodies[i].vel.x;
                    }
                    if ((bodies[i].pos.y - bodies[i].radius) < worldMinY)
                    {
                        bodies[i].pos.y = worldMinY + bodies[i].radius;
                        bodies[i].vel.y = -e_wall * bodies[i].vel.y;
                    }
                    if ((bodies[i].pos.y + bodies[i].radius) > worldMaxY)
                    {
                        bodies[i].pos.y = worldMaxY - bodies[i].radius;
                        bodies[i].vel.y = -e_wall * bodies[i].vel.y;
                    }
                }

                // membrane handling: check attempted crossing and allow/deny appropriately
                for (int i = 0; i < N; ++i)
                {
                    bool attempted_cross = false;
                    if ((prev_pos[i].x < membrane_x && bodies[i].pos.x >= membrane_x) ||
                        (prev_pos[i].x > membrane_x && bodies[i].pos.x <= membrane_x))
                    {
                        attempted_cross = true;
                    }

                    if (!attempted_cross)
                    {
                        // no crossing attempt this step; nothing to do
                        continue;
                    }

                    // check if impact Y is inside any pore (pores defined in world coords)
                    double yimpact = bodies[i].pos.y;
                    bool in_pore = false;
                    for (const auto &p : pores)
                    {
                        if (yimpact >= p.first && yimpact <= p.second) { in_pore = true; break; }
                    }

                    if (pores_open && in_pore)
                    {
                        // pores open: solvent allowed; solute blocked
                        if (is_solute[i])
                        {
                            // block solute: reflect and place just outside membrane on origin side
                            if (prev_pos[i].x < membrane_x)
                                bodies[i].pos.x = membrane_x - bodies[i].radius - 1e-6;
                            else
                                bodies[i].pos.x = membrane_x + bodies[i].radius + 1e-6;
                            bodies[i].vel.x = -e_wall * bodies[i].vel.x;
                        }
                        else
                        {
                            // solvent allowed --> ensure it's fully beyond the membrane on the destination side
                            if (prev_pos[i].x < membrane_x)
                                bodies[i].pos.x = membrane_x + bodies[i].radius + 1e-6; // move to just past membrane
                            else
                                bodies[i].pos.x = membrane_x - bodies[i].radius - 1e-6;
                            // leave velocity largely unchanged for flow
                        }
                    }
                    else
                    {
                        // membrane closed OR attempted crossing outside pores: block everything
                        if (prev_pos[i].x < membrane_x)
                            bodies[i].pos.x = membrane_x - bodies[i].radius - 1e-6;
                        else
                            bodies[i].pos.x = membrane_x + bodies[i].radius + 1e-6;
                        bodies[i].vel.x = -e_wall * bodies[i].vel.x;
                    }
                }

                // particle-particle collisions (pairwise)
                for (int i = 0; i < N; ++i)
                {
                    for (int j = i + 1; j < N; ++j)
                    {
                        Vec r = bodies[j].pos - bodies[i].pos;
                        double d = norm(r);
                        double minDist = bodies[i].radius + bodies[j].radius;
                        if (d < minDist && d > 1e-9)
                        {
                            Vec n = r / d;
                            Vec rel = bodies[i].vel - bodies[j].vel;
                            double relN = dot(rel, n);
                            double mi = bodies[i].mass;
                            double mj = bodies[j].mass;
                            double J = -(1 + e_obj) * relN / (1.0/mi + 1.0/mj);
                            bodies[i].vel += n * (J/mi);
                            bodies[j].vel -= n * (J/mj);

                            // separate them to avoid overlap
                            double overlap = 0.5 * (minDist - d);
                            bodies[i].pos -= n * overlap;
                            bodies[j].pos += n * overlap;
                        }
                    }
                }

                simTime += dt;
                simAcc -= 1.0;
            }
        }

        // --- Rendering ---
        window.clear(sf::Color::Black);

        // draw box
        window.draw(boxShape);

        // draw membrane bar (solid)
        sf::RectangleShape membraneRect(sf::Vector2f((float)membrane_thickness, (float)worldH));
        membraneRect.setOrigin((float)(membrane_thickness*0.5f), 0.0f);
        membraneRect.setPosition((float)membrane_x, (float)worldMinY);
        membraneRect.setFillColor(sf::Color(180,180,180));
        window.draw(membraneRect);

        // draw pores visually only when open (rectangles centered at membrane_x)
        if (pores_open)
        {
            for (const auto &p : pores)
            {
                sf::RectangleShape poreRect(sf::Vector2f((float)pore_width, (float)(p.second - p.first)));
                poreRect.setOrigin((float)(pore_width*0.5f), 0.0f);
                poreRect.setPosition((float)membrane_x, (float)p.first);
                poreRect.setFillColor(sf::Color(50,200,50,200));
                window.draw(poreRect);
            }
        }

        // draw particles
        for (int i = 0; i < N; ++i)
        {
            if (!bodies[i].alive) continue;
            sf::CircleShape c((float)bodies[i].radius);
            c.setOrigin((float)bodies[i].radius, (float)bodies[i].radius);
            c.setPosition((float)bodies[i].pos.x, (float)bodies[i].pos.y);
            c.setFillColor(bodies[i].color);
            window.draw(c);
        }

        // velocity vectors optional
        if (showVectors)
        {
            double vecScale = 10.0;
            for (int i = 0; i < N; ++i)
            {
                if (!bodies[i].alive) continue;
                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f((float)bodies[i].pos.x, (float)bodies[i].pos.y)),
                    sf::Vertex(sf::Vector2f((float)(bodies[i].pos.x + vecScale*bodies[i].vel.x), (float)(bodies[i].pos.y + vecScale*bodies[i].vel.y)))
                };
                window.draw(line, 2, sf::Lines);
            }
        }

        // HUD
        static sf::Font font;
        static bool fontLoaded = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf");
        if (fontLoaded)
        {
            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(13);
            t.setFillColor(sf::Color::White);
            std::string info = "SPACE=play/pause  O=open/close membrane  R=reset  V=toggle vectors  Up/Down=speed  ";
            info += "Time: " + std::to_string(simTime).substr(0,6);
            t.setString(info);
            t.setPosition(10.0f, 10.0f);
            window.draw(t);
        }

        window.display();
    }

    return 0;
}
