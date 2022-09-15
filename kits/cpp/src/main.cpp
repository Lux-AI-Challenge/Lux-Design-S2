#include <iostream>

#include "lux/json.hpp"

int main() {
    while (true) {
        json input;
        std::cin >> input;

        json output = json::parse(R"({"faction": "AlphaStrike", "bid": 10})");
        std::cout << output << std::endl;
    }
    return 0;
}
