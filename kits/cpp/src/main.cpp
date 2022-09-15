#include <iostream>

#include "lux/json.hpp"
#include "lux/log.hpp"

int main() {
    while (true) {
        json input;
        std::cin >> input;

        LOG("this is a test");

        json output = json::parse(R"({"faction": "AlphaStrike", "bid": 10})");
        std::cout << output << std::endl;
    }
    return 0;
}
