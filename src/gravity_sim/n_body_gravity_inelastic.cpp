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
    else if ((sim_init_type == SimulationInitType::EQUAL_SIDED_TRIANGLE) && N == 3)
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
    // ---------- Two-galaxy merger (rotating disks, prograde) ----------
    // ---------- TWO GALAXIES: Cold rotating exponential disks + bulge ----------
    {
        const int N1 = N / 2;
        const int N2 = N - N1;

        // ★ Beautiful 1:1 spiral–spiral collision
        const double M1 = 5e5;
        const double M2 = 5e5;

        const double sep = 600.0;   // larger separation → cleaner first approach
        const double impact = 80.0; // slight impact parameter → long tidal tails
        const double softR_visual = 0.6;

        const double G_local = UniversalConstants::G_scaled;

        // Parabolic-like initial orbit
        const double v_rel = std::sqrt(2.0 * G_local * (M1 + M2) / sep);

        // Centers
        Vec center_screen((double)winW * 0.5, (double)winH * 0.5);
        Vec centerA = center_screen + Vec(-sep * 0.5, -impact * 0.5);
        Vec centerB = center_screen + Vec(+sep * 0.5, +impact * 0.5);

        // ★ Prograde–prograde velocities (best tidal tails)
        //    Slight tilt for visual asymmetry
        Vec vA(+0.58 * v_rel, -0.04 * v_rel);
        Vec vB(-0.58 * v_rel, +0.04 * v_rel);

        std::mt19937_64 rng(1234567);
        std::uniform_real_distribution<double> uni01(0.0, 1.0);
        std::normal_distribution<double> gauss0(0.0, 1.0);

        // ★ Updated disk parameters
        const double RdA = 55.0; // disk scale length
        const double RdB = 55.0;

        const double diskFracA = 0.90;
        const double diskFracB = 0.92; // slightly colder B-disk for asymmetry

        // Bulge parameters
        const double bulgeMassFraction = 0.18;
        const double bulgeScale = 10.0;

        init_bodies.clear();
        init_bodies.resize(N1 + N2);

        auto sample_exponential_radius = [&](double Rd, double rmin, double rmax) -> double
        {
            double u = uni01(rng);
            double factor = 1.0 - std::exp(-(rmax - rmin) / Rd);
            return -Rd * std::log(1.0 - u * factor) + rmin;
        };

        auto enclosed_mass_diskbulge = [&](double r, double Mtot, double Rd, double Mbulge) -> double
        {
            double Mdisk = Mtot - Mbulge;

            double x = r / Rd;
            double Md_enc = Mdisk * (1.0 - (1.0 + x) * std::exp(-x));

            double rb = bulgeScale;
            double Mb_enc = Mbulge * (r * r) / ((r + rb) * (r + rb) + 1e-9);

            double Menc = Md_enc + Mb_enc;
            if (Menc < 1.0)
                Menc = 1.0;

            return Menc;
        };

        // ---------------------------
        // Galaxy A
        // ---------------------------
        double MbulgeA = M1 * bulgeMassFraction;

        for (int i = 0; i < N1; ++i)
        {
            double u = uni01(rng);
            bool isDisk = (u < diskFracA);

            double r, theta;
            if (isDisk)
            {
                r = sample_exponential_radius(RdA, 2.0, 300.0);
                theta = 2 * M_PI * uni01(rng);
            }
            else
            {
                double U = uni01(rng);
                double denom = std::pow(U, -2.0 / 3.0) - 1.0;
                r = (denom <= 0.0) ? 0.0 : 12.0 / std::sqrt(denom);
                theta = 2 * M_PI * uni01(rng);
            }

            double x_local = r * std::cos(theta);
            double y_local = r * std::sin(theta);

            init_bodies[i].pos = centerA + Vec(x_local, y_local);
            init_bodies[i].mass = M1 / double(N1);

            double Menc = enclosed_mass_diskbulge(r, M1, RdA, MbulgeA);
            double vc = std::sqrt(G_local * Menc / (r + 1e-9));

            Vec tang(-std::sin(theta), std::cos(theta));

            double dispFrac = isDisk ? 0.05 : 0.12;
            Vec vel_disp(gauss0(rng) * dispFrac * vc,
                         gauss0(rng) * dispFrac * vc);

            init_bodies[i].vel = vA + tang * vc + vel_disp;

            init_bodies[i].radius = softR_visual;
            init_bodies[i].color = sf::Color(255, 140, 40);
            init_bodies[i].alive = true;
        }

        // ---------------------------
        // Galaxy B
        // ---------------------------
        double MbulgeB = M2 * bulgeMassFraction;

        for (int j = 0; j < N2; ++j)
        {
            int idx = N1 + j;
            double u = uni01(rng);
            bool isDisk = (u < diskFracB);

            double r, theta;
            if (isDisk)
            {
                r = sample_exponential_radius(RdB, 2.0, 300.0);
                theta = 2 * M_PI * uni01(rng);
            }
            else
            {
                double U = uni01(rng);
                double denom = std::pow(U, -2.0 / 3.0) - 1.0;
                r = (denom <= 0.0) ? 0.0 : 10.0 / std::sqrt(denom);
                theta = 2 * M_PI * uni01(rng);
            }

            double x_local = r * std::cos(theta);
            double y_local = r * std::sin(theta);

            init_bodies[idx].pos = centerB + Vec(x_local, y_local);
            init_bodies[idx].mass = M2 / double(N2);

            double Menc = enclosed_mass_diskbulge(r, M2, RdB, MbulgeB);
            double vc = std::sqrt(G_local * Menc / (r + 1e-9));

            Vec tang(-std::sin(theta), std::cos(theta));

            double dispFrac = isDisk ? 0.02 : 0.12;
            Vec vel_disp(gauss0(rng) * dispFrac * vc,
                         gauss0(rng) * dispFrac * vc);

            init_bodies[idx].vel = vB + tang * vc + vel_disp;

            init_bodies[idx].radius = softR_visual;
            init_bodies[idx].color = sf::Color(255, 140, 40);
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

// --------------------------------------------------------------
// Compute softened accelerations (pairwise)
// --------------------------------------------------------------
void computeAccelerations(
    std::vector<Body> &bodies,
    std::vector<Vec> &acc,
    double eps,
    bool galaxyCollisionMode,
    double G)
{
    size_t N = bodies.size();
    for (auto &a : acc)
        a = Vec(0, 0);

    for (size_t i = 0; i < N; ++i)
    {
        if (!bodies[i].alive)
            continue;

        for (size_t j = i + 1; j < N; ++j)
        {
            if (!bodies[j].alive)
                continue;

            Vec r = bodies[j].pos - bodies[i].pos;
            double d2 = r.x * r.x + r.y * r.y;

            // collision merging OFF for galaxy mode
            if (!galaxyCollisionMode)
            {
                double d = std::sqrt(d2);
                double minD = bodies[i].radius + bodies[j].radius;

                if (d < minD)
                {
                    double Mi = bodies[i].mass;
                    double Mj = bodies[j].mass;
                    double M = Mi + Mj;

                    // COM merge
                    bodies[i].pos = (bodies[i].pos * Mi + bodies[j].pos * Mj) / M;
                    bodies[i].vel = (bodies[i].vel * Mi + bodies[j].vel * Mj) / M;
                    bodies[i].mass = M;

                    bodies[j].alive = false;
                    continue;
                }
            }

            // softened
            double dist2_soft = d2 + eps * eps;
            double invDist = 1.0 / std::sqrt(dist2_soft);
            double invDist3 = invDist * invDist * invDist;

            double ai = G * bodies[j].mass * invDist3;
            double aj = G * bodies[i].mass * invDist3;

            acc[i].x += r.x * ai;
            acc[i].y += r.y * ai;

            acc[j].x -= r.x * aj;
            acc[j].y -= r.y * aj;
        }
    }

    // for (auto &a : acc)
    // {
    //     double aMag = std::sqrt(a.x * a.x + a.y * a.y);
    //     double aMax = 500.0; // set depending on scale
    //     if (aMag > aMax)
    //         a *= (aMax / aMag);
    // }
}

// --------------------------------------------------------------
// LEAPFROG INTEGRATOR (kick-drift-kick)
// --------------------------------------------------------------
void leapfrogStep(
    std::vector<Body> &bodies,
    std::vector<Vec> &acc,
    double dt,
    double eps,
    bool galaxyCollisionMode,
    double G)
{
    size_t N = bodies.size();

    // --- KICK 1: v += a * dt/2 ---
    for (size_t i = 0; i < N; ++i)
    {
        if (!bodies[i].alive)
            continue;
        bodies[i].vel += acc[i] * (0.5 * dt);
    }

    // --- DRIFT: x += v * dt ---
    for (size_t i = 0; i < N; ++i)
    {
        if (!bodies[i].alive)
            continue;
        bodies[i].pos += bodies[i].vel * dt;
    }

    // recompute accelerations at new positions
    computeAccelerations(bodies, acc, eps, galaxyCollisionMode, G);

    // --- KICK 2: v += a * dt/2 ---
    for (size_t i = 0; i < N; ++i)
    {
        if (!bodies[i].alive)
            continue;
        bodies[i].vel += acc[i] * (0.5 * dt);
    }
}

double computeKineticEnergy(const std::vector<Body> &bodies)
{
    double KE = 0.0;
    for (const auto &b : bodies)
    {
        if (!b.alive)
            continue;
        double v2 = b.vel.x * b.vel.x + b.vel.y * b.vel.y;
        KE += 0.5 * b.mass * v2;
    }
    return KE;
}

double computePotentialEnergy(
    const std::vector<Body> &bodies,
    double eps,
    double G)
{
    double PE = 0.0;
    size_t N = bodies.size();
    double eps2 = eps * eps;

    for (size_t i = 0; i < N; ++i)
    {
        if (!bodies[i].alive)
            continue;

        for (size_t j = i + 1; j < N; ++j)
        {
            if (!bodies[j].alive)
                continue;

            Vec r = bodies[j].pos - bodies[i].pos;
            double d2 = r.x * r.x + r.y * r.y + eps2;
            double dist = std::sqrt(d2);

            // Plummer-softened PE = -G m1 m2 / sqrt(r^2 + eps^2)
            PE += -G * bodies[i].mass * bodies[j].mass / dist;
        }
    }

    return PE;
}

int main()
{
    Config cfg = ConfigLoader::load("/root/projects/N_Body_Problem/simulation_configs/n_body_gravity_configs.json");
    const double dt = cfg.sim.time_step;
    // simulation speed: how many fixed `dt` steps to run per frame (can be fractional)
    double simSpeed = cfg.sim.default_sim_speed; // default: 20 dt-steps per frame (faster-than-real-time)
    double simStepsAcc = 0.0;                    // accumulator for fractional steps
    double softeningLength = cfg.sim.softening;  // for smoothing simSpeed changes
    bool galaxyCollisionMode = (cfg.sim.sim_init_type == SimulationInitType::TWO_GALAXIES_COLLISION);
    const int winW = cfg.window.width;
    const int winH = cfg.window.height;

    double totalEnergy = 0.0;
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;

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

    sf::Font font;
    if (font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    {
        // font loaded
    }
    else
    {
        std::cerr << "Error loading font\n";
        return -1;
    }

    // ---- LEAPFROG INITIAL HALF-KICK ----
    std::vector<Vec> acc(bodies.size(), Vec(0, 0));
    computeAccelerations(bodies, acc, softeningLength, galaxyCollisionMode, UniversalConstants::G_scaled);

    // Perform initial half-kick
    for (size_t i = 0; i < bodies.size(); ++i)
    {
        if (!bodies[i].alive)
            continue;
        bodies[i].vel += acc[i] * (0.5 * dt);
    }

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
                    init_system(init_bodies, N, winW, winH, cfg.sim.sim_init_type);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::M)
                {
                    // decrease N and reinitialize
                    N = std::max(2, N - 1);
                    init_system(init_bodies, N, winW, winH, cfg.sim.sim_init_type);
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
                // run one leapfrog step
                leapfrogStep(
                    bodies,
                    acc,
                    dt,
                    softeningLength,
                    galaxyCollisionMode,
                    UniversalConstants::G_scaled);

                // handle trails
                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    if (!bodies[i].alive)
                        continue;

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

                // ---- ENERGY CALCULATION ----
                kineticEnergy = computeKineticEnergy(bodies);
                potentialEnergy = computePotentialEnergy(bodies, softeningLength, UniversalConstants::G_scaled);
                totalEnergy = kineticEnergy + potentialEnergy;
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

        sf::Text txt;
        txt.setFont(font);
        txt.setCharacterSize(14);
        txt.setFillColor(sf::Color::White);
        std::string info = "Space: pause/play | R: reset | N: +1 Body | M: -1 Body |  V: toggle vectors  |  T: toggle trails \n";
        info += "Sim time: " + std::to_string(simTime).substr(0, 6);
        info += "  |  Speed: " + std::to_string(simSpeed).substr(0, 6);
        info += "  |  N: " + std::to_string((int)bodies.size());
        info += "\nKE: " + std::to_string(kineticEnergy).substr(0, 10);
        info += "  |  PE: " + std::to_string(potentialEnergy).substr(0, 10);
        info += "  |  Total E: " + std::to_string(totalEnergy).substr(0, 10);
        txt.setString(info);
        txt.setPosition(8, 8);
        window.draw(txt);

        window.display();
    }

    return 0;
}
