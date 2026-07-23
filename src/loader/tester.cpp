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
              // std::ifstream file(path, std::ios::binary);
    // const auto json_size = read_u64_le(file);
    // std::cout << "json_size: " << json_size << std::endl;
    // std::vector<char> data( json_size + 1, '\0');
    // file.read(data.data(), json_size);
    // std::cout << data.data() << "\n";
    return 0;


}