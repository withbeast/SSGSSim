project (main LANGUAGES CXX CUDA)
enable_language(CUDA)

add_executable(main main.cu lif.cu lif.cuh)