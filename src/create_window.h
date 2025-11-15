#pragma once
#include <SFML/Graphics.hpp>
#include <SFML/Window.hpp>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <string>

void CreateWindow(sf::RenderWindow& window, int winW, int winH, const std::string& window_name)
{
    // Create hidden window to avoid jumping before centering
    window.create(sf::VideoMode(winW, winH), window_name, sf::Style::Default);
    //window.setVisible(false);

    // --- Open X11 display ---
    Display* display = XOpenDisplay(nullptr);
    if (!display) {
        // Fallback: couldn't open X11 display → just center using SFML DesktopMode
        auto desktop = sf::VideoMode::getDesktopMode();
        int posX = (desktop.width  - winW) / 2;
        int posY = (desktop.height - winH) / 2;
        window.setPosition({posX, posY});
        window.setVisible(true);
        return;
    }

    // Try to get _NET_WORKAREA
    Atom workAreaAtom = XInternAtom(display, "_NET_WORKAREA", True);

    int posX = 0, posY = 0;

    if (workAreaAtom != None)
    {
        Atom actualType;
        int actualFormat;
        unsigned long count, bytes;
        unsigned char* data = nullptr;

        if (Success == XGetWindowProperty(
            display, DefaultRootWindow(display),
            workAreaAtom, 0, 4, False,
            XA_CARDINAL, &actualType, &actualFormat,
            &count, &bytes, &data)
            && data != nullptr)
        {
            long* work = (long*)data;

            int wx = work[0];
            int wy = work[1];
            int ww = work[2];
            int wh = work[3];

            posX = wx + (ww - winW) / 2;
            posY = wy + (wh - winH) / 2;

            XFree(data);
        }
        else
        {
            // Fallback: just use desktop size
            auto desktop = sf::VideoMode::getDesktopMode();
            posX = (desktop.width  - winW) / 2;
            posY = (desktop.height - winH) / 2;
        }
    }
    else
    {
        // WSLg: _NET_WORKAREA missing → fallback
        auto desktop = sf::VideoMode::getDesktopMode();
        posX = (desktop.width  - winW) / 2;
        posY = (desktop.height - winH) / 2;
    }

    XCloseDisplay(display);

    window.setPosition({posX, posY});
    window.setVisible(true);
    window.setFramerateLimit(120);
}
