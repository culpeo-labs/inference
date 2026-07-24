#include "loader/loader.h"
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
    const auto t = culpeo::loader::safe_tensors::load({ path });
    std::cout << "Count: " << t.tensor_count() << std::endl;
    for (auto & name : t.keys())
    {
        std::cout << name << "\n";
    }
    return 0;


}