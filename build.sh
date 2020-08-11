# `g++ ./tests/build_tests.cpp -o ./tests/build_tests`
# `g++ ./tests/base_benchmark.cpp -o ./tests/base_benchmark -Ofast`
# `g++ ./tests/threads/pthread.c -o ./tests/threads/pthread -pthread`
# `g++ ./tests/error.cpp -o ./tests/error -Ofast -march=native -pthread`
# `g++ ./tests/fast_benchmark.cpp -o ./tests/fast_benchmark -Ofast -march=native -pthread`
`g++ ./matrix/matrix_test.cpp -o ./matrix/matrix_test -Ofast -march=native`
`g++ ./layer/layer_test.cpp -o ./layer/layer_test -Ofast -march=native`
`g++ ./model/model_test.cpp -o ./model/model_test -Ofast -march=native`