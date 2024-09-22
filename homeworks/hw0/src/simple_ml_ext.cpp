#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // 临时变量
    std::vector<float> logits(batch * k);   // 用于存储每个 batch 的 logits
    std::vector<float> softmax_probs(batch * k);  // 存储 softmax 概率
    std::vector<float> gradient(n * k, 0);  // 梯度初始化为0

    for (size_t i = 0; i < m; i += batch) {
        size_t current_batch_size = std::min(batch, m - i);

        // Step 1: 计算 logits, X[i:i+batch] @ theta
        for (size_t b = 0; b < current_batch_size; ++b) {
            for (size_t j = 0; j < k; ++j) {
                logits[b * k + j] = 0;
                for (size_t l = 0; l < n; ++l) {
                    logits[b * k + j] += X[(i + b) * n + l] * theta[l * k + j];
                }
            }
        }

        // Step 2: 计算 softmax 概率（先减去每行的最大值）
        for (size_t b = 0; b < current_batch_size; ++b) {
            float max_logit = *std::max_element(&logits[b * k], &logits[b * k + k]);

            // 计算 exp(logits - max_logit) 以数值稳定
            float sum_exp = 0;
            for (size_t j = 0; j < k; ++j) {
                softmax_probs[b * k + j] = std::exp(logits[b * k + j] - max_logit);
                sum_exp += softmax_probs[b * k + j];
            }

            // 归一化 softmax 概率
            for (size_t j = 0; j < k; ++j) {
                softmax_probs[b * k + j] /= sum_exp;
            }
        }

        // Step 3: 计算梯度
        std::fill(gradient.begin(), gradient.end(), 0);  // 初始化梯度
        for (size_t b = 0; b < current_batch_size; ++b) {
            for (size_t j = 0; j < k; ++j) {
                // 计算 (softmax_probs - I_y)
                float indicator = (y[i + b] == j) ? 1.0f : 0.0f;
                float diff = softmax_probs[b * k + j] - indicator;

                // 累积梯度: X_batch.T @ (softmax_probs - I_y)
                for (size_t l = 0; l < n; ++l) {
                    gradient[l * k + j] += X[(i + b) * n + l] * diff;
                }
            }
        }

        // Step 4: 用 SGD 更新 theta
        for (size_t l = 0; l < n; ++l) {
            for (size_t j = 0; j < k; ++j) {
                theta[l * k + j] -= lr * gradient[l * k + j] / current_batch_size;
            }
        }
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
