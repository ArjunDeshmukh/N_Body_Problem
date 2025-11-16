#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#include "create_window.h"
#include "body.h"
#include "vector_math.h"
#include "universal_constants.h"
#include "config.h"

// Initialize a system of N bodies. If N==3, optionally create the figure-eight choreography;
// otherwise generate a random distribution of masses/positions/velocities.
static void init_system(std::vector<Body> &init_bodies, int N, int winW, int winH, SimulationInitType sim_init_type = SimulationInitType::FIGURE_EIGHT_CHOREOGRAPHY)
{
    init_bodies.clear();
    init_bodies.resize(N);

    if ((sim_init_type == SimulationInitType::STABLE_ORBITAL_SYSTEM) && N == 2)
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
        double v_circ = std::sqrt(UniversalConstants::G_scaled * init_bodies[0].mass / orbitR);
        // perpendicular velocity (counter-clockwise)
        init_bodies[1].vel = {0.0, -v_circ};
        init_bodies[1].radius = 8.0;
        init_bodies[1].color = sf::Color::Blue;
        init_bodies[1].alive = true;

        // conserve momentum (give small recoil to central mass so total momentum = 0)
        init_bodies[0].vel.x = -(init_bodies[1].mass * init_bodies[1].vel.x) / init_bodies[0].mass;
        init_bodies[0].vel.y = -(init_bodies[1].mass * init_bodies[1].vel.y) / init_bodies[0].mass;
        
    }
    else if((sim_init_type == SimulationInitType::EQUAL_SIDED_TRIANGLE) && N == 3)
    {
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
        
    }
    else if (sim_init_type == SimulationInitType::FIGURE_EIGHT_CHOREOGRAPHY)
    {
        // Place N bodies along a figure-eight (Gerono lemniscate) curve and
        // set velocities tangent to the curve so the system starts as a choreography-like setup.
        const double PI = std::acos(-1.0);
        const double centerX = winW * 0.5;
        const double centerY = winH * 0.5;
        const double L_scale = std::min(winW, winH) * 0.32; // spatial scale
        const double mass_each = 1000.0;
        const double vel_scale = std::sqrt(UniversalConstants::G_scaled * mass_each / L_scale);

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
    else if (sim_init_type == SimulationInitType::TWO_GALAXIES_COLLISION)
    {
         // Parameters (tune these)
        const int N1 = N/2;         // number of particles in galaxy A
        const int N2 = N - N/2;          // number of particles in galaxy B
        const double M1 = 5e5;      // total mass of galaxy A
        const double M2 = 2e5;      // total mass of galaxy B
        const double sep = 500.0;   // initial center separation along x
        const double impact = 50.0; // initial y offset (gives non-zero impact parameter)
        const double softR = 4.0;   // visual/softening base radius

        // Derived velocity scales (approx)
        const double G_local = UniversalConstants::G_scaled; // use your defined G_scaled
        const double v_escape = std::sqrt(2.0 * G_local * (M1 + M2) / sep);
        const double orbit_factor = 0.70; // 1.0 = parabolic, <1 = bound (elliptical), >1 = hyperbolic
        const double v_rel = orbit_factor * v_escape;

        // centers and COM velocities (place A left, B right)
        Vec centerA((double)winW * 0.5 - sep * 0.5, (double)winH * 0.5 - impact * 0.5);
        Vec centerB((double)winW * 0.5 + sep * 0.5, (double)winH * 0.5 + impact * 0.5);

        // set COM velocities so relative speed is v_rel, and total momentum = 0
        // Give B a leftward + small upward velocity, A opposite to conserve momentum
        Vec vB(-v_rel * (M1 / (M1 + M2)), +0.08 * v_rel);
        Vec vA(v_rel * (M2 / (M1 + M2)), -0.08 * v_rel);

        // Fill particles: simple Plummer-ish radial distribution + small velocity dispersion
        std::mt19937_64 rng(1234567);
        std::uniform_real_distribution<double> uni(0.0, 1.0);
        auto plummer_radius_sample = [&](double a) -> double
        {
            // sample cumulative for Plummer: r = a * (u^{-2/3} - 1)^{-1/2}
            double u = uni(rng);
            double denom = std::pow(u, -2.0 / 3.0) - 1.0;
            if (denom <= 0.0)
                return 0.0;
            return a / std::sqrt(denom);
        };

        // clear and prepare
        init_bodies.clear();
        init_bodies.resize(N1 + N2);

        // Galaxy A
        double aA = 60.0; // Plummer scale length (visual)
        for (int i = 0; i < N1; ++i)
        {
            double r = plummer_radius_sample(aA);
            double theta = 2.0 * M_PI * uni(rng);
            Vec offset(r * std::cos(theta), r * std::sin(theta));
            init_bodies[i].pos = centerA + offset;
            init_bodies[i].mass = M1 / double(N1);
            // particle velocity = COM velocity + small dispersion (kept small so system is initially bound)
            double vel_disp = std::sqrt(G_local * (M1 / 10.0) / (aA + 1.0)); // heuristic
            init_bodies[i].vel = vA + Vec((uni(rng) - 0.5) * vel_disp, (uni(rng) - 0.5) * vel_disp);
            // init_bodies[i].radius = softR * std::pow(init_bodies[i].mass, 0.33) * 0.5;
            init_bodies[i].radius = 0.1;
            init_bodies[i].color = sf::Color::Red;
            init_bodies[i].alive = true;
        }

        // Galaxy B
        double aB = 50.0;
        for (int j = 0; j < N2; ++j)
        {
            int idx = N1 + j;
            double r = plummer_radius_sample(aB);
            double theta = 2.0 * M_PI * uni(rng);
            Vec offset(r * std::cos(theta), r * std::sin(theta));
            init_bodies[idx].pos = centerB + offset;
            init_bodies[idx].mass = M2 / double(N2);
            double vel_disp = std::sqrt(G_local * (M2 / 10.0) / (aB + 1.0));
            init_bodies[idx].vel = vB + Vec((uni(rng) - 0.5) * vel_disp, (uni(rng) - 0.5) * vel_disp);
            // init_bodies[idx].radius = softR * std::pow(init_bodies[idx].mass, 0.33) * 0.5;
            init_bodies[idx].radius = 0.1;
            init_bodies[idx].color = sf::Color::Blue;
            init_bodies[idx].alive = true;
        }

        return;

    }
    else
    {
        // Random distribution
        std::mt19937_64 rng(1234567);
        std::uniform_real_distribution<double> uniPosX(50.0, (double)(winW - 50));
        std::uniform_real_distribution<double> uniPosY(50.0, (double)(winH - 50));
        std::uniform_real_distribution<double> uniVel(-5.0, 5.0);
        std::uniform_real_distribution<double> uniMass(5.0, 500.0);
        std::uniform_real_distribution<double> uniRadius(2.0, 8.0);
        std::uniform_int_distribution<int> uniColor(0, 255);

        for (int i = 0; i < N; ++i)
        {
            double m = uniMass(rng);
            double radius = uniRadius(rng);
            init_bodies[i].mass = m;
            init_bodies[i].pos = Vec(uniPosX(rng), uniPosY(rng));
            init_bodies[i].vel = Vec(uniVel(rng), uniVel(rng));
            init_bodies[i].radius = radius;
            init_bodies[i].color = sf::Color(uniColor(rng), uniColor(rng), uniColor(rng));
            init_bodies[i].alive = true;
        }
    }
}

int main()
{
    Config cfg = ConfigLoader::load("/root/projects/N_Body_Problem/simulation_configs/n_body_gravity_configs.json");
    const double dt = cfg.sim.time_step;
    // simulation speed: how many fixed `dt` steps to run per frame (can be fractional)
    double simSpeed = cfg.sim.default_sim_speed; // default: 20 dt-steps per frame (faster-than-real-time)
    double simStepsAcc = 0.0;                    // accumulator for fractional steps
    const int winW = cfg.window.width;
    const int winH = cfg.window.height;

    // Create window
    sf::RenderWindow window;
    CreateWindow(window, winW, winH, "N-Body Gravity: Perfectly Inelastic Collisions");

    // --- Initial conditions (N-body) ---
    int N = cfg.sim.n_particles; // default number of bodies
    std::cout << "[Main] Initializing system with N = " << N << " bodies.\n";

    std::vector<Body> bodies;
    std::vector<Body> init_bodies;
    init_system(init_bodies, N, winW, winH, cfg.sim.sim_init_type);
    bodies = init_bodies;
    std::vector<sf::VertexArray> trails(N, sf::VertexArray(sf::LinesStrip));

    bool running = true;
    bool showVectors = false;
    bool showTrails = false;

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
                {
                    // Toggle velocity vectors
                    showVectors = !showVectors;
                }
                if (ev.key.code == sf::Keyboard::T)
                {
                    // Toggle trails
                    showTrails = !showTrails;
                }
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
                        double a_i = UniversalConstants::G_scaled * bodies[j].mass / (d * d);
                        double a_j = UniversalConstants::G_scaled * bodies[i].mass / (d * d);

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
        
        if (showTrails)
        {
            for (auto &t : trails)
                if (t.getVertexCount() > 1)
                    window.draw(t);
        }

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
            std::string info = "Space: pause/play | R: reset | N: +1 Body | M: -1 Body |  V: toggle vectors  |  T: toggle trails \n";
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
