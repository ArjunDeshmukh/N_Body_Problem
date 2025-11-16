#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <fstream>
#include <iostream>
#include "json.hpp"

using json = nlohmann::json;

struct WindowConfig
{
    int width = 1200;
    int height = 800;
    bool fullscreen = false;
};

enum SimulationInitType : uint8_t
{
    STABLE_ORBITAL_SYSTEM = 0, // Only for 2 body system with gravity
    FIGURE_EIGHT_CHOREOGRAPHY = 1,
    EQUAL_SIDED_TRIANGLE = 2, // Only for 3 body system with gravity
    TWO_GALAXIES_COLLISION = 3
};

struct SimConfig
{
    int n_particles = 10;
    double time_step = 0.01;
    double softening = 0.001;
    double default_sim_speed = 20.0;
    bool collisions = true;
    SimulationInitType sim_init_type = SimulationInitType::FIGURE_EIGHT_CHOREOGRAPHY;
};

struct Config
{
    WindowConfig window;
    SimConfig sim;
};

// ---------------------------
// CONFIG LOADER
// ---------------------------

class ConfigLoader
{
public:
    static inline SimulationInitType parseSimInitType(const std::string &s)
    {
        if (s == "STABLE_ORBITAL_SYSTEM")
            return STABLE_ORBITAL_SYSTEM;
        if (s == "FIGURE_EIGHT_CHOREOGRAPHY")
            return FIGURE_EIGHT_CHOREOGRAPHY;
        if (s == "EQUAL_SIDED_TRIANGLE")
            return EQUAL_SIDED_TRIANGLE;
        if (s == "TWO_GALAXIES_COLLISION")
            return TWO_GALAXIES_COLLISION;

        std::cerr << "[ConfigLoader] Unknown sim_init_type string: " << s
                  << ". Using default.\n";
        return FIGURE_EIGHT_CHOREOGRAPHY;
    }

    static Config load(const std::string &path)
    {
        Config cfg;
        json j;

        // Read file
        std::ifstream file(path);
        if (!file)
        {
            std::cerr << "[ConfigLoader] Could not open " << path << ". Using defaults.\n";
            return cfg;
        }

        try
        {
            file >> j;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ConfigLoader] JSON parse error: " << e.what()
                      << ". Using defaults.\n";
            return cfg;
        }

        std::cout << "[ConfigLoader] Loaded configuration from " << path << ".\n";

        // Window section
        if (j.contains("window"))
        {
            json &w = j["window"];
            cfg.window.width = w.value("width", cfg.window.width);
            cfg.window.height = w.value("height", cfg.window.height);
            cfg.window.fullscreen = w.value("fullscreen", cfg.window.fullscreen);
        }

        // Simulation section
        if (j.contains("simulation"))
        {
            json &s = j["simulation"];
            cfg.sim.n_particles = s.value("n_particles", cfg.sim.n_particles);
            cfg.sim.time_step = s.value("time_step", cfg.sim.time_step);
            cfg.sim.softening = s.value("softening", cfg.sim.softening);
            cfg.sim.default_sim_speed = s.value("default_sim_speed", cfg.sim.default_sim_speed);
            cfg.sim.collisions = s.value("collisions", cfg.sim.collisions);

            if (s.contains("sim_init_type"))
            {
                std::string typeStr = s["sim_init_type"].get<std::string>();
                cfg.sim.sim_init_type = parseSimInitType(typeStr);
            }
        }

        return cfg;
    }
};

#endif // CONFIG_HPP
