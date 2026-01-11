// topple_client.hpp
#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>

#if __has_include(<tf2_geometry_msgs/tf2_geometry_msgs.hpp>)
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <nlohmann/json.hpp>

namespace topple {

struct ToppleConfig {
    double edge_m = 0.05;
    double dir_x = 1.0;
    double dir_y = 0.0;
    double friction = 0.6;
    double restitution = 0.0;
    double mass = 0.2;
    std::string mode = "force";  // "force"|"torque"|"velocity"
    bool gui = true;
    std::string python = "python3";
    std::string script;          // e.g., "<share>/scripts/topple_sim_cli.py"
    std::string package_name = "blocksworld_executor";
};

inline std::string resolve_script_path(const ToppleConfig& cfg) {
    if (!cfg.script.empty())
        return cfg.script;

    const auto share = ament_index_cpp::get_package_share_directory(
        cfg.package_name.empty() ? std::string("blocksworld_executor") : cfg.package_name);

    return share + "/scripts/topple_sim_cli.py";
}

inline std::string build_request_json(
    const std::vector<std::string>& ids,
    const std::vector<geometry_msgs::msg::Pose>& poses,
    const std::string& topple_id,
    const ToppleConfig& cfg)
{
    using nlohmann::json;
    if (ids.size() != poses.size()) {
        throw std::runtime_error("ids.size() != poses.size()");
    }
    json j;
    j["edge"] = cfg.edge_m;
    j["direction"] = {cfg.dir_x, cfg.dir_y};
    j["friction"] = cfg.friction;
    j["restitution"] = cfg.restitution;
    j["mass"] = cfg.mass;
    j["mode"] = cfg.mode;
    j["gui"] = cfg.gui;
    j["topple_id"] = topple_id;
    auto& arr = j["cubes"];
    arr = json::array();
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto& P = poses[i];
        json c;
        c["id"] = ids[i];
        c["position"] = {P.position.x, P.position.y, P.position.z};
        c["orientation"] = {P.orientation.x, P.orientation.y, P.orientation.z, P.orientation.w};
        arr.push_back(std::move(c));
    }
    return j.dump();
}

inline std::string run_python_with_stdin(const std::string& python, const std::string& script, const std::string& payload)
{
    // Build a command that reads JSON from stdin and prints JSON on stdout
    std::string cmd = python + " " + script; // no args -> reads stdin
    FILE* pipe = popen(cmd.c_str(), "w");
    if (!pipe) throw std::runtime_error("Failed to start Python process for writing.");

    // Write the JSON request to the child's stdin
    fwrite(payload.data(), 1, payload.size(), pipe);
    pclose(pipe);

    // Now run the script again but this time capture stdout in one call:
    // (Simplest approach: write payload to a temp file, then call --json tmp)
    // For atomicity and simplicity, use tmpfile():
    char tmpPath[] = "/tmp/topple_req_XXXXXX.json";
    int fd = mkstemps(tmpPath, 5); // ".json" suffix length=5
    if (fd == -1) throw std::runtime_error("mkstemps failed");
    FILE* f = fdopen(fd, "w");
    fwrite(payload.data(), 1, payload.size(), f);
    fclose(f);

    std::string cmd2 = python + " " + script + " --json " + tmpPath + " -v";
    FILE* out = popen(cmd2.c_str(), "r");
    if (!out) {
        std::remove(tmpPath);
        throw std::runtime_error("Failed to start Python process for reading.");
    }
    std::ostringstream ss;
    char buffer[4096];
    size_t n;
    while ((n = fread(buffer, 1, sizeof(buffer), out)) > 0) {
        ss.write(buffer, n);
    }
    pclose(out);
    std::remove(tmpPath);
    return ss.str();
}

inline std::string run_python_with_json_file(
    const std::string& python,
    const std::string& script,
    const std::string& payload)
{
    char in_template[]  = "/tmp/topple_req_XXXXXX.json";
    char out_template[] = "/tmp/topple_out_XXXXXX.json";
    int fd_in = mkstemps(in_template, 5);  // ".json"
    int fd_out = mkstemps(out_template, 5);
    if (fd_in == -1 || fd_out == -1)
        throw std::runtime_error("mkstemps failed");

    FILE* f = fdopen(fd_in, "w");
    if (!f) {
        close(fd_in);
        throw std::runtime_error("fdopen failed");
    }
    fwrite(payload.data(), 1, payload.size(), f);
    fclose(f);

    std::string cmd = python + " " + script + " --json " + in_template + " -v" + " --out " + out_template;
    int rc = system(cmd.c_str());
    if (rc != 0) {
        std::remove(in_template);
        std::remove(out_template);
        throw std::runtime_error("Python sim exited with non-zero status");
    }

    // read the JSON result
    std::ifstream fin(out_template);
    if (!fin) {
        std::remove(in_template);
        std::remove(out_template);
        throw std::runtime_error("Failed to open sim output file");
    }
    std::ostringstream ss;
    ss << fin.rdbuf();
    std::string out = ss.str();

    std::remove(in_template);
    std::remove(out_template);
    return out;
}


inline std::vector<geometry_msgs::msg::Pose>
simulate_topple_with_pybullet(
    const std::vector<std::string>& ids,
    const std::vector<geometry_msgs::msg::Pose>& poses,
    const std::string& topple_id,
    const ToppleConfig& cfg = ToppleConfig())
{
    using nlohmann::json;
    // 1) Build JSON request
    const std::string script = resolve_script_path(cfg);
    std::string req = build_request_json(ids, poses, topple_id, cfg);

    // 2) Execute Python script and capture JSON response
    //std::string out = run_python_with_stdin(cfg.python, script, req);
    std::string out = run_python_with_json_file(cfg.python, script, req);

    // 3) Parse JSON
    json j = json::parse(out);
    const auto& arr = j.at("poses");

    // 4) Return poses in the SAME order as input IDs
    //    (Python already preserves input order; we’ll map by id just in case.)
    std::unordered_map<std::string, geometry_msgs::msg::Pose> map;
    map.reserve(arr.size());
    for (const auto& c : arr) {
        geometry_msgs::msg::Pose P;
        auto p = c.at("position");
        auto q = c.at("orientation");
        P.position.x = p[0]; P.position.y = p[1]; P.position.z = p[2];
        P.orientation.x = q[0]; P.orientation.y = q[1]; P.orientation.z = q[2]; P.orientation.w = q[3];
        map.emplace(c.at("id").get<std::string>(), P);
    }
    std::vector<geometry_msgs::msg::Pose> out_poses;
    out_poses.reserve(ids.size());
    for (const auto& id : ids) {
        auto it = map.find(id);
        if (it == map.end()) throw std::runtime_error("Missing id in response: " + id);
        out_poses.push_back(it->second);
    }
    return out_poses;
}

} // namespace topple
