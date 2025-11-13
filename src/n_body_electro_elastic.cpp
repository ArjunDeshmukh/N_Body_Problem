#include <SFML/Graphics.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

// N-body electrostatic simulation (all bodies negative charge)
// Elastic collisions between bodies (bounce)

struct Vec {
    double x, y;
    Vec(double _x = 0, double _y = 0) : x(_x), y(_y) {}
    Vec operator+(const Vec &o) const { return Vec(x + o.x, y + o.y); }
    Vec operator-(const Vec &o) const { return Vec(x - o.x, y - o.y); }
    Vec operator*(double s) const { return Vec(x * s, y * s); }
    Vec operator/(double s) const { return Vec(x / s, y / s); }
    Vec &operator+=(const Vec &o) { x += o.x; y += o.y; return *this; }
};

double norm(const Vec &v) { return std::sqrt(v.x * v.x + v.y * v.y); }
double dot(const Vec &a, const Vec &b) { return a.x*b.x + a.y*b.y; }

struct Body {
    double mass;
    double q; // electric charge (negative)
    Vec pos;
    Vec vel;
    double radius;
    sf::Color color;
    bool alive = true;
};

// Coulomb constant: 1 / (4 * pi * epsilon0)
// epsilon0 (vacuum permittivity) = 8.854187817e-12 F/m
// K = 1/(4*pi*epsilon0)
const double EPS0 = 8.854187817e-12;
const double K = 1.0 / (4.0 * M_PI * EPS0);
// visual scaling factor applied to initial velocities to make the simulation
// visually interesting (dimensionless). Tweak as needed.
// Tuned value to produce visible motion without immediate ejection.
// Increased to make initial motion visibly faster.
const double VISUAL_VEL_SCALE = 1e7;
// Maximum velocity (pixels per second) to clamp after integration to avoid runaway speeds
const double MAX_VEL = 1e7;
// Cap for initial velocities assigned at system init (pixels/second)
const double INITIAL_VEL_CAP = 600.0;
// physical constants for H+ (proton)
const double PROTON_MASS = 1.67262192369e-27; // kg
const double ELEMENTARY_CHARGE = 1.602176634e-19; // C

static void clamp_initial_velocities(std::vector<Body> &bodies)
{
    for (auto &b : bodies)
    {
        if (!b.alive) continue;
        if (!std::isfinite(b.vel.x) || !std::isfinite(b.vel.y)) { b.vel = Vec(0.0, 0.0); continue; }
        double v = norm(b.vel);
        if (v > INITIAL_VEL_CAP && v > 0.0)
        {
            b.vel = b.vel * (INITIAL_VEL_CAP / v);
        }
    }
}


static void init_system(std::vector<Body> &init_bodies, int N, int winW, int winH)
{
    init_bodies.clear();
    init_bodies.resize(N);

    if (N == 2)
    {
        // central massive negatively charged body (use proton mass and elementary charge)
        init_bodies[0].mass = PROTON_MASS;
        init_bodies[0].q = -ELEMENTARY_CHARGE;
        init_bodies[0].pos = {(double)(winW * 0.5), (double)(winH * 0.5)};
        init_bodies[0].vel = {0.0, 0.0};
        init_bodies[0].radius = 22.0;
        init_bodies[0].color = sf::Color::Red;
        init_bodies[0].alive = true;

        // satellite
        const double orbitR = 200.0;
        init_bodies[1].mass = PROTON_MASS;
        init_bodies[1].q = -ELEMENTARY_CHARGE;
        init_bodies[1].pos = {init_bodies[0].pos.x + orbitR, init_bodies[0].pos.y};
        // approximate circular speed from central attraction/repulsion magnitude
        double v_circ = std::sqrt(std::fabs(K * init_bodies[0].q * init_bodies[1].q) / (init_bodies[1].mass * orbitR));
        v_circ *= VISUAL_VEL_SCALE;
        init_bodies[1].vel = {0.0, -v_circ};
        init_bodies[1].radius = 8.0;
        init_bodies[1].color = sf::Color::Blue;
        init_bodies[1].alive = true;

        // give small recoil to central mass so total momentum = 0
        init_bodies[0].vel.x = -(init_bodies[1].mass * init_bodies[1].vel.x) / init_bodies[0].mass;
        init_bodies[0].vel.y = -(init_bodies[1].mass * init_bodies[1].vel.y) / init_bodies[0].mass;
        // clamp initial velocities to avoid enormous startup speeds
        clamp_initial_velocities(init_bodies);
        return;
    }

    if (N == 3)
    {
        // use the same figure-eight choreography positions, but assign negative charge
        const double centerX = winW * 0.5;
        const double centerY = winH * 0.5;
        const double L_scale = 80.0;
        const double mass_each = PROTON_MASS;
        const Vec r1_unit(0.97000436, -0.24308753);
        const Vec r2_unit(-0.97000436, 0.24308753);
        const Vec r3_unit(0.0, 0.0);
        const Vec v1_unit(0.4662036850, 0.4323657300);
        const Vec v2_unit(0.4662036850, 0.4323657300);
        const Vec v3_unit(-0.93240737, -0.86473146);
        const double vel_scale = std::sqrt((K * std::fabs(ELEMENTARY_CHARGE)) / L_scale) * VISUAL_VEL_SCALE;

        init_bodies[0] = {mass_each, -ELEMENTARY_CHARGE, {(double)(centerX + r1_unit.x * L_scale), (double)(centerY + r1_unit.y * L_scale)}, {(double)(v1_unit.x * vel_scale), (double)(v1_unit.y * vel_scale)}, 10.0, sf::Color::Red, true};
        init_bodies[1] = {mass_each, -ELEMENTARY_CHARGE, {(double)(centerX + r2_unit.x * L_scale), (double)(centerY + r2_unit.y * L_scale)}, {(double)(v2_unit.x * vel_scale), (double)(v2_unit.y * vel_scale)}, 10.0, sf::Color::Blue, true};
        init_bodies[2] = {mass_each, -ELEMENTARY_CHARGE, {(double)(centerX + r3_unit.x * L_scale), (double)(centerY + r3_unit.y * L_scale)}, {(double)(v3_unit.x * vel_scale), (double)(v3_unit.y * vel_scale)}, 10.0, sf::Color::Green, true};
        // clamp initial velocities
        clamp_initial_velocities(init_bodies);
        return;
    }

    // general N: distribute bodies along a figure-eight and give negative charge
    const double PI = std::acos(-1.0);
    const double centerX = winW * 0.5;
    const double centerY = winH * 0.5;
    const double L_scale = std::min(winW, winH) * 0.32;
    // use proton mass and elementary charge (negative sign for electron-like negative species)
    const double mass_each = PROTON_MASS;
    const double q_scale = -ELEMENTARY_CHARGE; // negative charge (Coulombs)
    const double vel_scale = std::sqrt((K * std::fabs(q_scale)) / L_scale) * VISUAL_VEL_SCALE;

    const sf::Color palette[6] = {sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Yellow, sf::Color::Magenta, sf::Color::Cyan};

    for (int i = 0; i < N; ++i)
    {
        double t = 2.0 * PI * double(i) / double(N);
        double xu = std::sin(t);
        double yu = 0.5 * std::sin(2.0 * t);
        double dx = std::cos(t);
        double dy = std::cos(2.0 * t);

        double m = mass_each;
        double radius = std::cbrt(m) * 0.6 + 3.0;
        init_bodies[i].mass = m;
        init_bodies[i].q = q_scale;
        init_bodies[i].pos = Vec(centerX + xu * L_scale, centerY + yu * L_scale);
        init_bodies[i].vel = Vec(dx * vel_scale, dy * vel_scale);
        init_bodies[i].radius = radius;
        init_bodies[i].color = palette[i % 6];
        init_bodies[i].alive = true;
    }
    // clamp initial velocities for general N
    clamp_initial_velocities(init_bodies);
}

// Ensure bodies are inside the axis-aligned box centered at `center` with half-sizes
static void clamp_positions_to_box(std::vector<Body> &bodies, const Vec &center, double boxHalfW, double boxHalfH)
{
    double minX = center.x - boxHalfW;
    double maxX = center.x + boxHalfW;
    double minY = center.y - boxHalfH;
    double maxY = center.y + boxHalfH;
    for (auto &b : bodies)
    {
        if (!b.alive) continue;
        if (!std::isfinite(b.pos.x) || !std::isfinite(b.pos.y)) continue;
        double left = minX + b.radius;
        double right = maxX - b.radius;
        double top = minY + b.radius;
        double bottom = maxY - b.radius;
        if (b.pos.x < left) b.pos.x = left;
        if (b.pos.x > right) b.pos.x = right;
        if (b.pos.y < top) b.pos.y = top;
        if (b.pos.y > bottom) b.pos.y = bottom;
    }
}


int main()
{
    const double dt = 0.01;
    double simSpeed = 20.0;
    double simStepsAcc = 0.0;
    const int winW = 1000;
    const int winH = 800;

    sf::RenderWindow window(sf::VideoMode(winW, winH), "N-Body Electrostatic: Elastic Collisions");
    window.setFramerateLimit(120);

    // World box half-sizes (in pixels). The visual box remains fixed on screen,
    // but wall collisions are applied to world positions mapped to the current view center.
    const double boxHalfW = 400.0;
    const double boxHalfH = 300.0;

    // Create a screen-space drawable rectangle (fixed on screen)
    sf::RectangleShape boxShape(sf::Vector2f((float)(boxHalfW * 2.0), (float)(boxHalfH * 2.0)));
    boxShape.setPosition((float)((winW * 0.5) - boxHalfW), (float)((winH * 0.5) - boxHalfH));
    boxShape.setFillColor(sf::Color::Transparent);
    boxShape.setOutlineColor(sf::Color(200,200,200));
    boxShape.setOutlineThickness(2.0f);

    int N = 10;
    std::vector<Body> bodies;
    std::vector<Body> init_bodies;
    init_system(init_bodies, N, winW, winH);
    bodies = init_bodies;
    // clamp initial positions to the box centered on the initial center-of-mass
    Vec cm_init(0,0);
    double totalM_init = 0.0;
    for (const auto &b : bodies) { if (b.alive) { cm_init += b.pos * b.mass; totalM_init += b.mass; } }
    if (totalM_init > 0.0) cm_init = cm_init / totalM_init;
    clamp_positions_to_box(bodies, cm_init, boxHalfW, boxHalfH);
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
                    init_system(init_bodies, N, winW, winH);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    // clamp init positions to the current center-of-mass
                    Vec cm_new(0,0); double tm = 0;
                    for (const auto &b : bodies) { if (b.alive) { cm_new += b.pos * b.mass; tm += b.mass; } }
                    if (tm > 0) cm_new = cm_new / tm;
                    clamp_positions_to_box(bodies, cm_new, boxHalfW, boxHalfH);
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::M)
                {
                    N = std::max(2, N - 1);
                    init_system(init_bodies, N, winW, winH);
                    bodies = init_bodies;
                    trails.assign(N, sf::VertexArray(sf::LinesStrip));
                    Vec cm_new2(0,0); double tm2 = 0;
                    for (const auto &b : bodies) { if (b.alive) { cm_new2 += b.pos * b.mass; tm2 += b.mass; } }
                    if (tm2 > 0) cm_new2 = cm_new2 / tm2;
                    clamp_positions_to_box(bodies, cm_new2, boxHalfW, boxHalfH);
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::R)
                {
                    bodies = init_bodies;
                    // clamp reset positions to the box
                    Vec cm_rst(0,0); double tm_rst = 0;
                    for (const auto &b : bodies) { if (b.alive) { cm_rst += b.pos * b.mass; tm_rst += b.mass; } }
                    if (tm_rst > 0) cm_rst = cm_rst / tm_rst;
                    clamp_positions_to_box(bodies, cm_rst, boxHalfW, boxHalfH);
                    for (auto &t : trails) t.clear();
                    simTime = 0.0;
                }
                if (ev.key.code == sf::Keyboard::V)
                    showVectors = !showVectors;
            }
        }

        // center camera on center of mass BEFORE physics so world-box collisions
        // are computed consistently relative to the camera that will be used.
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
        if (totalMass > 0) centerOfMass = centerOfMass / totalMass;
        view.setCenter((float)centerOfMass.x, (float)centerOfMass.y);
        window.setView(view);

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

                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    if (!bodies[i].alive) continue;
                    for (size_t j = i + 1; j < bodies.size(); ++j)
                    {
                        if (!bodies[j].alive) continue;

                        Vec r = bodies[j].pos - bodies[i].pos;
                        double d = norm(r);
                        // avoid singularity / huge forces at very small separations
                        if (d < 1e-12) {
                            // extremely close — treat as tiny separation to avoid div0
                            d = 1e-12;
                        }
                        double minDist = bodies[i].radius + bodies[j].radius;

                        // Coulomb acceleration (repulsion/attraction). No collision handling;
                        // let electrostatic forces (repulsion since charges are same-sign) govern motion.
                        Vec dir = r / d;
                        // Coulomb acceleration: a_i += -K * q_i * q_j * dir / (d^2 * m_i)
                        double qq = bodies[i].q * bodies[j].q;
                        double factor = K * qq / (d * d);
                        acc[i] += dir * (-factor / bodies[i].mass);
                        acc[j] += dir * (+factor / bodies[j].mass);
                    }
                }

                // integrate and handle elastic collisions with the axis-aligned box
                for (size_t i = 0; i < bodies.size(); ++i)
                {
                    if (!bodies[i].alive) continue;
                    bodies[i].vel += acc[i] * dt;
                    bodies[i].pos += bodies[i].vel * dt;

                    // wall collisions: fully elastic (reverse perpendicular velocity)
                    if (bodies[i].pos.x - bodies[i].radius < worldBoxMinX)
                    {
                        bodies[i].pos.x = worldBoxMinX + bodies[i].radius;
                        bodies[i].vel.x = -bodies[i].vel.x;
                    }
                    else if (bodies[i].pos.x + bodies[i].radius > worldBoxMaxX)
                    {
                        bodies[i].pos.x = worldBoxMaxX - bodies[i].radius;
                        bodies[i].vel.x = -bodies[i].vel.x;
                    }

                    if (bodies[i].pos.y - bodies[i].radius < worldBoxMinY)
                    {
                        bodies[i].pos.y = worldBoxMinY + bodies[i].radius;
                        bodies[i].vel.y = -bodies[i].vel.y;
                    }
                    else if (bodies[i].pos.y + bodies[i].radius > worldBoxMaxY)
                    {
                        bodies[i].pos.y = worldBoxMaxY - bodies[i].radius;
                        bodies[i].vel.y = -bodies[i].vel.y;
                    }

                    if (trails[i].getVertexCount() > 500)
                    {
                        sf::VertexArray temp(sf::LinesStrip);
                        for (size_t k = 1; k < trails[i].getVertexCount(); ++k) temp.append(trails[i][k]);
                        trails[i] = temp;
                    }
                }

                simTime += dt;
                simStepsAcc -= 1.0;
            }
        }

        

        window.clear(sf::Color::Black);

        for (auto &t : trails) if (t.getVertexCount() > 1) window.draw(t);

        for (const auto &b : bodies)
        {
            if (!b.alive) continue;
            sf::CircleShape c((float)b.radius);
            c.setOrigin((float)b.radius, (float)b.radius);
            c.setPosition((float)b.pos.x, (float)b.pos.y);
            c.setFillColor(b.color);
            window.draw(c);

            if (showVectors)
            {
                sf::Vertex line[] = { sf::Vertex(sf::Vector2f((float)b.pos.x, (float)b.pos.y)), sf::Vertex(sf::Vector2f((float)(b.pos.x + b.vel.x), (float)(b.pos.y + b.vel.y))) };
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
            txt.setString(info);
            txt.setPosition(8, 8);
            window.draw(txt);
        }

        window.display();
    }

    return 0;
}
