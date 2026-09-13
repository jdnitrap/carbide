// main.cpp — entry point: carbide [--data <path>]
// Mirrors carbide_modules/__main__.py's startup order: parse --data, load
// the dataset once, then enter the main menu loop.
//
// Build:      g++ -std=c++17 -O2 -pthread main.cpp -o carbide
// Run:        ./carbide [--data file.txt]
// Verify:     g++ -std=c++17 -O2 gradcheck_test.cpp -o gradcheck_test && ./gradcheck_test
//             g++ -std=c++17 -O2 model_smoke_test.cpp -o model_smoke_test && ./model_smoke_test
#include "cli.hpp"
#include "interrupt.hpp"
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    carbide::install_sigint_handler();

    std::string data_arg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_arg = argv[i + 1];
            ++i;
        }
    }

    if (!data_arg.empty()) carbide::load_dataset(data_arg);
    else carbide::load_dataset();

    carbide::main_loop();
    return 0;
}
