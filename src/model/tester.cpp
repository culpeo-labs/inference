#include "model/safetensors.h"
#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

// std::uint64_t read_u64_le(std::ifstream & file)
// {
//     std::uint64_t value;
//     file.read(reinterpret_cast<char *>(&value), sizeof(value));
//     return value;
// }

int main(int, char **)
{
    constexpr std::string_view path{ "/workspaces/inference/model.safetensors"};
    const auto t = culpeo::inference::model::safetensors::load({ path });
    if (!t.has_value())
    {
        std::cerr << t.error() << "\n";
        return 1;
    }
    const auto model_file = *t;
    std::cout << "Count: " << model_file.tensor_count() << std::endl;
    for (auto & name : model_file.keys())
    {
        std::cout << name << "\n";
    }
    return 0;


}