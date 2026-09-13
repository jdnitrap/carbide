// cli.hpp — mirrors carbide_modules/cli.py: all nine menu_*() handlers plus
// main(). The same training_active guards added to the Python version to
// fix the model/opt/data race are carried forward here from the start.
//
// One honest scope cut from the Python version: there is no matplotlib
// here (this build has zero external dependencies by design), so
// menu_save() and run_ablation() write CSV instead of PNG plots.
#pragma once
#include "training.hpp"
#include "generation.hpp"
#include "checkpoint.hpp"
#include "interrupt.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <random>
#include <optional>
#include <iomanip>

namespace carbide {

inline std::string read_line(const std::string& prompt) {
    std::cout << prompt;
    std::string line;
    if (!std::getline(std::cin, line)) return "";
    // trim
    size_t a = line.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = line.find_last_not_of(" \t\r\n");
    return line.substr(a, b - a + 1);
}

inline std::optional<long long> try_parse_int(const std::string& s) {
    try {
        size_t pos;
        long long v = std::stoll(s, &pos);
        if (pos != s.size()) return std::nullopt;
        return v;
    } catch (...) { return std::nullopt; }
}

inline std::optional<double> try_parse_double(const std::string& s) {
    try {
        size_t pos;
        double v = std::stod(s, &pos);
        if (pos != s.size()) return std::nullopt;
        return v;
    } catch (...) { return std::nullopt; }
}

// forward decls for the training-menu helpers
inline void reset_and_train();
inline void resume_training();

inline void menu_main() {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CARBIDE INTERACTIVE CLI\n";
    std::cout << "Dataset: " << config.data_file << " (" << data.size() << " bytes)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << R"(
MAIN MENU:
  1. Train model
  2. REPL (generate text interactively)
  3. Hyperparameters
  4. Checkpoints
  5. Inspect MDBE embeddings
  6. Run ablation study
  7. Load dataset
  8. Save outputs
  9. Exit
)";
}

inline void menu_train() {
    bool is_running = train_state.training_active;
    long long cur_step, cur_target; double cur_loss;
    {
        std::lock_guard<std::mutex> lk(train_state.training_lock);
        cur_step = train_state.step;
        cur_target = train_state.training_target_steps;
        cur_loss = train_state.current_loss;
    }

    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "TRAINING MENU" << (is_running ? " [RUNNING IN BACKGROUND]" : "") << "\n";
    std::cout << std::string(70, '-') << "\n";
    printf("Current: step %lld/%lld, loss %.4f\n\n", cur_step, cur_target, cur_loss);

    if (is_running) {
        std::cout << R"(
  1. Show training log (Option 1)
  2. Show live monitor (Option 2 - btop style)
  3. Show both (Option 3)
  4. Stop background training
  5. Pause (save checkpoint)
  6. Back to main menu (training continues)
)";
    } else {
        std::cout << R"(
  1. Start training in background (Exit menu, training runs)
  2. Start training and watch log (Standard output)
  3. Start training and watch live monitor (btop-style)
  4. Start training and watch both (Log + btop)
  5. Resume from checkpoint
  6. Reset and train from scratch
  7. Back to main menu
)";
    }

    std::string choice = read_line("Choice: ");

    if (is_running) {
        if (choice == "1") show_training_log();
        else if (choice == "2") show_live_monitor();
        else if (choice == "3") show_both_monitors();
        else if (choice == "4") stop_background_training();
        else if (choice == "5") { save_checkpoint("_paused"); std::cout << "  \xE2\x9C\x93 Checkpoint saved\n"; }
        else if (choice == "6") {
            std::cout << "  (Training continues in background. Returning to main menu...)\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } else {
        auto n = try_parse_int(read_line("  Steps to run: "));
        if (!n) { std::cout << "  \xE2\x9C\x97 Please enter a number\n"; return; }
        train_state.training_target_steps = train_state.step + *n;

        if (choice == "1") {
            start_background_training(*n, "none");
            std::cout << "  \xE2\x9C\x93 Training started in background (" << *n << " steps)\n";
            std::cout << "  Returning to main menu...\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else if (choice == "2") start_background_training(*n, "log");
        else if (choice == "3") start_background_training(*n, "monitor");
        else if (choice == "4") start_background_training(*n, "both");
        else if (choice == "5") resume_training();
        else if (choice == "6") reset_and_train();
    }
}

inline void reset_and_train() {
    auto n = try_parse_int(read_line("  Steps to train: "));
    if (!n) { std::cout << "  \xE2\x9C\x97 Please enter a number\n"; return; }
    train_state.reset();
    model.reset();
    opt.reset();
    train_n_steps(*n);
}

inline void resume_training() {
    std::string suffix = read_line("  Checkpoint suffix (default blank): ");
    if (load_checkpoint(suffix.empty() ? "" : "_" + suffix)) {
        auto n = try_parse_int(read_line("  Additional steps to run: "));
        if (!n) { std::cout << "  \xE2\x9C\x97 Please enter a number\n"; return; }
        train_n_steps(*n);
    }
}

inline void menu_repl() {
    std::cout << "\n" << std::string(70, '-') << "\nREPL MODE \xE2\x80\x94 Type prompts to generate continuations\n"
              << std::string(70, '-') << "\n";
    std::cout << "Commands: :help, :temp <0.5-2.0>, :topk <5-50>, :exit\n\n";

    double temperature = 0.8;
    int top_k = 20;

    while (true) {
        std::string prompt = read_line("> ");
        if (prompt.empty()) continue;
        if (prompt == ":exit") break;
        if (prompt == ":help") {
            std::cout << "  :temp <val>  - set temperature (higher=more random)\n";
            std::cout << "  :topk <val>  - set top-k filtering\n";
            std::cout << "  :exit        - back to menu\n";
            continue;
        }
        if (prompt.rfind(":temp ", 0) == 0) {
            auto v = try_parse_double(prompt.substr(6));
            if (v) { temperature = *v; std::cout << "  \xE2\x9C\x93 Temperature set to " << temperature << "\n"; }
            else std::cout << "  Error: invalid value\n";
            continue;
        }
        if (prompt.rfind(":topk ", 0) == 0) {
            auto v = try_parse_int(prompt.substr(6));
            if (v) { top_k = (int)*v; std::cout << "  \xE2\x9C\x93 Top-k set to " << top_k << "\n"; }
            else std::cout << "  Error: invalid value\n";
            continue;
        }
        try {
            std::string cont = generate(prompt, 80, temperature, top_k);
            std::cout << "  " << cont << "\n\n";
        } catch (const std::exception& e) {
            std::cout << "  Error: " << e.what() << "\n";
        }
    }
}

inline void menu_hyperparams() {
    std::cout << "\n" << std::string(70, '-') << "\nHYPERPARAMETERS\n" << std::string(70, '-') << "\n";
    printf(R"(
Current settings:
  d_model:      %d
  n_layers:     %d
  d_state:      %d
  batch_size:   %d
  learning_rate: %g

1. d_model (embedding dimension)
2. n_layers (SSM blocks)
3. d_state (state dimension per SSM)
4. batch_size
5. learning_rate
6. Back
)", config.d_model, config.n_layers, config.d_state, config.batch_size, config.learning_rate);

    std::string choice = read_line("Edit: ");

    if ((choice == "1" || choice == "2" || choice == "3") && train_state.training_active) {
        std::cout << "  \xE2\x9A\xA0 Background training is active \xE2\x80\x94 stop it first (Main Menu \xE2\x86\x92 1 \xE2\x86\x92 Stop background training)\n";
        std::cout << "    before changing d_model/n_layers/d_state. Batch size and learning rate are safe to\n";
        std::cout << "    change while training runs.\n";
        return;
    }

    if (choice == "1") {
        auto v = try_parse_int(read_line("  New d_model: "));
        if (!v) { std::cout << "  \xE2\x9C\x97 Please enter a valid number\n"; return; }
        config.d_model = (int)*v;
    } else if (choice == "2") {
        auto v = try_parse_int(read_line("  New n_layers: "));
        if (!v) { std::cout << "  \xE2\x9C\x97 Please enter a valid number\n"; return; }
        config.n_layers = (int)*v;
    } else if (choice == "3") {
        auto v = try_parse_int(read_line("  New d_state: "));
        if (!v) { std::cout << "  \xE2\x9C\x97 Please enter a valid number\n"; return; }
        config.d_state = (int)*v;
    } else if (choice == "4") {
        auto v = try_parse_int(read_line("  New batch_size: "));
        if (!v) { std::cout << "  \xE2\x9C\x97 Please enter a valid number\n"; return; }
        config.batch_size = (int)*v;
    } else if (choice == "5") {
        auto v = try_parse_double(read_line("  New learning_rate: "));
        if (!v) { std::cout << "  \xE2\x9C\x97 Please enter a valid number\n"; return; }
        config.learning_rate = *v;
    }

    if (choice == "1" || choice == "2" || choice == "3") {
        model.reset();
        opt.reset();
        std::cout << "  \xE2\x9A\xA0 Model structure changed \xE2\x80\x94 model will be reinitialized on next training step\n";
    }
    if (choice == "4" || choice == "5") std::cout << "  \xE2\x9C\x93 Updated (takes effect on next training)\n";
}

inline void menu_checkpoints() {
    std::cout << "\n" << std::string(70, '-') << "\nCHECKPOINTS\n" << std::string(70, '-') << "\n";

    std::vector<std::string> ckpts;
    if (std::filesystem::exists(config.checkpoint_dir)) {
        for (auto& e : std::filesystem::directory_iterator(config.checkpoint_dir)) {
            auto name = e.path().filename().string();
            if (name.size() > 4 && name.substr(name.size() - 4) == ".bin")
                ckpts.push_back(name.substr(0, name.size() - 4));
        }
    }
    if (!ckpts.empty()) {
        std::cout << "  Available checkpoints:\n";
        for (auto& c : ckpts) std::cout << "    " << c << "\n";
    } else {
        std::cout << "  No checkpoints found\n";
    }

    std::cout << R"(
1. Save checkpoint
2. Load checkpoint
3. List checkpoints
4. Back
)";
    std::string choice = read_line("Choice: ");

    if (choice == "1") {
        std::string suffix = read_line("  Suffix (e.g., 'final'): ");
        save_checkpoint(suffix.empty() ? "" : "_" + suffix);
    } else if (choice == "2") {
        if (train_state.training_active) {
            std::cout << "  \xE2\x9A\xA0 Background training is active \xE2\x80\x94 stop it first (Main Menu \xE2\x86\x92 1 \xE2\x86\x92 Stop background training)\n";
            std::cout << "    before loading a checkpoint. Saving is fine while training runs.\n";
            return;
        }
        std::string suffix = read_line("  Suffix to load: ");
        load_checkpoint(suffix.empty() ? "" : "_" + suffix);
    }
    // choice 3: already listed above
}

inline void menu_mdbe() {
    std::cout << "\n" << std::string(70, '-') << "\nMDBE EMBEDDINGS\n" << std::string(70, '-') << "\n";
    if (!model) { std::cout << "  Model not initialized\n"; return; }

    std::vector<std::pair<int, std::string>> sample = {{32, "SPACE"}, {65, "A"}, {97, "a"}, {48, "0"}, {33, "!"}};
    std::cout << "\n  Sample embeddings (first 8 dims):\n\n";
    printf("  %6s %-10s Embedding dims\n", "Byte", "Char");
    std::cout << "  " << std::string(60, '-') << "\n";
    for (auto& [b, ch] : sample) {
        printf("  %6d %-10s", b, ch.c_str());
        for (int d = 0; d < 8 && d < config.d_model; ++d)
            printf(" %7.3f", model->mdbe.base->data[(size_t)b * config.d_model + d]);
        printf("\n");
    }
}

inline void run_ablation() {
    std::cout << "\n  Running ablation (3 \xC3\x97 1500 steps)...\n\n";

    auto run_mode = [&](Mode mode, const std::string& name, int steps = 1500, unsigned seed = 42) {
        std::mt19937 r(seed);
        Carbide model_v(config.d_model, config.n_layers, config.d_state, r);
        AdamW opt_v(model_v.parameters(), config.learning_rate);
        std::vector<double> curve;
        curve.reserve(steps);
        for (int s = 1; s <= steps; ++s) {
            auto batch = get_batch();
            auto logits = model_v.forward(batch.xs, batch.B, batch.T, mode);
            auto loss = ag::cross_entropy(logits, batch.ys);
            opt_v.zero_grad();
            ag::backward_scalar(loss);
            auto params = model_v.parameters();
            clip_grad_norm(params, 1.0);
            opt_v.step();
            curve.push_back(loss->data[0]);
            if (s % 250 == 0) {
                double avg = 0; int cnt = std::min(100, (int)curve.size());
                for (int i = (int)curve.size() - cnt; i < (int)curve.size(); ++i) avg += curve[i];
                avg /= cnt;
                printf("    %-18s step %4d | loss %.4f\n", name.c_str(), s, avg);
            }
        }
        return curve;
    };

    std::cout << "  Training mode: full\n";
    auto full = run_mode(Mode::Full, "full");
    std::cout << "  Training mode: no_constraints\n";
    auto noc = run_mode(Mode::NoConstraints, "no_constraints");
    std::cout << "  Training mode: plain_embedding\n";
    auto plain = run_mode(Mode::PlainEmbedding, "plain_embedding");

    std::ofstream fh("carbide_ablation.csv");
    fh << "step,full,no_constraints,plain_embedding\n";
    for (size_t i = 0; i < full.size(); ++i)
        fh << (i + 1) << "," << full[i] << "," << noc[i] << "," << plain[i] << "\n";

    std::cout << "\n  \xE2\x9C\x93 Ablation complete: carbide_ablation.csv\n";
    std::cout << "  (this build has no plotting library \xE2\x80\x94 raw curves only, no PNG)\n";
}

inline void menu_ablation() {
    std::cout << "\n" << std::string(70, '-') << "\nABLATION STUDY\n" << std::string(70, '-') << "\n";
    std::cout << R"(
  Testing: full constraints vs no constraints vs plain embedding
  This will train 3 models for 1500 steps each.

1. Run full ablation
2. Cancel
)";
    if (read_line("Choice: ") == "1") run_ablation();
}

inline void augment_data_with_generation() {
    if (!model || train_state.step == 0) {
        std::cout << "  \xE2\x9A\xA0 Model not trained yet. Train first, then return here.\n";
        return;
    }
    std::cout << "\n  Generating text to augment dataset...\n";
    auto n_samples = try_parse_int(read_line("  How many samples to generate? (1-10): "));
    auto n_bytes_per = try_parse_int(read_line("  Bytes per sample? (50-500): "));
    if (!n_samples || !n_bytes_per) { std::cout << "  \xE2\x9C\x97 Please enter a number\n"; return; }

    std::mt19937 r(std::random_device{}());
    std::uniform_int_distribution<int> letter(97, 122);
    std::vector<std::string> generated;
    for (int i = 0; i < std::min((int)*n_samples, 10); ++i) {
        std::string prompt(1, (char)letter(r));
        std::string cont = generate(prompt, (int)*n_bytes_per, 0.8, 20);
        generated.push_back(cont);
        std::cout << "    Sample " << (i + 1) << ": " << cont.substr(0, 60) << "...\n";
    }

    std::vector<int> augmented = data;
    for (auto& text : generated) for (unsigned char c : text) augmented.push_back((int)c);
    data = augmented;
    config.data_file = "[augmented with generated text]";

    std::cout << "\n  \xE2\x9C\x93 Dataset augmented: " << augmented.size() << " bytes total\n";
    std::cout << "  You can now retrain the model on this augmented data.\n";
    std::cout << "  Main Menu \xE2\x86\x92 1 (Train) \xE2\x86\x92 Reset and train from scratch\n";
}

inline void menu_load_dataset() {
    std::cout << "\n" << std::string(70, '-') << "\nLOAD DATASET\n" << std::string(70, '-') << "\n";
    std::cout << "Current dataset: " << config.data_file << " (" << data.size() << " bytes)\n\n";

    if (train_state.training_active) {
        std::cout << "  \xE2\x9A\xA0 Background training is active \xE2\x80\x94 stop it first (Main Menu \xE2\x86\x92 1 \xE2\x86\x92 Stop background training)\n";
        std::cout << "    before loading a different dataset.\n";
        return;
    }

    std::vector<std::string> txt_files;
    for (auto& e : std::filesystem::directory_iterator("."))
        if (e.path().extension() == ".txt") txt_files.push_back(e.path().filename().string());

    if (!txt_files.empty()) {
        std::cout << "  \xF0\x9F\x93\x81 Available .txt files in current directory:\n\n";
        for (size_t i = 0; i < txt_files.size(); ++i) {
            auto sz = std::filesystem::file_size(txt_files[i]);
            printf("    [%zu] %-40s (%10lld bytes)\n", i + 1, txt_files[i].c_str(), (long long)sz);
        }
        std::cout << "\n";
    }

    std::cout << R"(
MENU:
  1. Select from list above (enter number)
  2. Type full path manually
  3. Use fallback sample
  4. Use train.txt (if exists)
  5. Generate text from model & retrain (Option 2)
  6. Back
)";
    std::string choice = read_line("Choice: ");

    if (choice == "1") {
        if (txt_files.empty()) { std::cout << "  \xE2\x9C\x97 No .txt files found in current directory\n"; return; }
        auto idx = try_parse_int(read_line("  Select file number: "));
        if (!idx) { std::cout << "  \xE2\x9C\x97 Please enter a number\n"; return; }
        long long i = *idx - 1;
        if (i >= 0 && i < (long long)txt_files.size()) load_dataset(txt_files[i]);
        else std::cout << "  \xE2\x9C\x97 Invalid selection\n";
    } else if (choice == "2") {
        std::cout << "\n  Enter full path to file:\n  Examples:\n    my_data.txt\n    /home/user/Documents/story.txt\n";
        std::string filepath = read_line("  Path: ");
        if (!filepath.empty()) load_dataset(filepath);
    } else if (choice == "3") {
        std::string text;
        for (int i = 0; i < 500; ++i) text += FALLBACK;
        data = bytes_of(text);
        config.data_file = "[fallback]";
        std::cout << "  \xE2\x9C\x93 Using fallback (" << data.size() << " bytes)\n";
    } else if (choice == "4") {
        if (std::filesystem::exists("train.txt")) load_dataset("train.txt");
        else std::cout << "  \xE2\x9C\x97 train.txt not found\n";
    } else if (choice == "5") {
        augment_data_with_generation();
    }
    // choice 6: back
}

inline void menu_save() {
    std::cout << "\n" << std::string(70, '-') << "\nSAVE OUTPUTS\n" << std::string(70, '-') << "\n";
    if (train_state.loss_history.empty()) { std::cout << "  \xE2\x9A\xA0 No training data to export\n"; return; }

    {
        std::ofstream fh("carbide_loss.csv");
        fh << "step,loss\n";
        for (auto& [s, l] : train_state.loss_history) fh << s << "," << l << "\n";
        std::cout << "  \xE2\x9C\x93 carbide_loss.csv\n";
        std::cout << "  (this build has no plotting library \xE2\x80\x94 no PNG output)\n";
    }

    if (model) {
        std::ofstream fh("mdbe_table.csv");
        fh << "byte,character";
        for (int i = 0; i < config.d_model; ++i) fh << ",cell_" << i;
        fh << ",is_alpha,is_digit,is_upper,is_punct,is_space,utf8_lead\n";
        for (int b = 0; b < VOCAB; ++b) {
            fh << b << ",";
            if (b >= 32 && b < 127) fh << "\"" << (char)b << "\"";
            fh << ",";
            for (int i = 0; i < config.d_model; ++i)
                fh << std::fixed << std::setprecision(4) << model->mdbe.base->data[(size_t)b * config.d_model + i] << ",";
            auto cols = mdbe_constraints({b}, true);
            for (int i = 0; i < NUM_CONSTRAINTS; ++i) fh << (int)cols->data[i] << (i + 1 < NUM_CONSTRAINTS ? "," : "");
            fh << "\n";
        }
        std::cout << "  \xE2\x9C\x93 mdbe_table.csv\n";
    }

    std::cout << "\n  All outputs saved.\n";
}

inline void main_loop() {
    std::cout << R"(
+--------------------------------------------------------------------------+
|                CARBIDE INTERACTIVE CLI (C++, zero dependencies)          |
|         Byte-Level SSM Language Model with REPL Interface                |
+--------------------------------------------------------------------------+
)";

    while (true) {
        menu_main();
        std::string choice = read_line("Choice: ");

        if (choice == "1") menu_train();
        else if (choice == "2") menu_repl();
        else if (choice == "3") menu_hyperparams();
        else if (choice == "4") menu_checkpoints();
        else if (choice == "5") menu_mdbe();
        else if (choice == "6") menu_ablation();
        else if (choice == "7") menu_load_dataset();
        else if (choice == "8") menu_save();
        else if (choice == "9") {
            if (train_state.training_active) {
                std::cout << "\n  Background training is still running \xE2\x80\x94 stopping it first...\n";
                stop_background_training();
            }
            // a still-joinable-but-finished thread would call std::terminate()
            // at process exit if never joined — cover that path too.
            if (train_state.training_thread.joinable()) train_state.training_thread.join();
            std::cout << "\n\xE2\x9C\x93 Goodbye!\n\n";
            break;
        } else {
            std::cout << "  \xE2\x9C\x97 Invalid choice\n";
        }
    }
}

} // namespace carbide
