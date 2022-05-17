
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightconv_cuda.cuh"

std::vector<at::Tensor> lightconv_cuda_forward(at::Tensor input, at::Tensor filters, int padding_l) {

    at::DeviceGuard g(input.device());
    const auto minibatch = input.size(0);
    const auto numFeatures = input.size(1);
    const auto sequenceLength = input.size(2);

    const auto numHeads = filters.size(0);
    const auto filterSize = filters.size(1);

    const auto numFiltersInBlock = numFeatures / numHeads;

    const dim3 blocks(minibatch, numFeatures);

    auto output = at::zeros_like(input);
    auto stream = at::cuda::getCurrentCUDAStream();

    if (sequenceLength <= 32) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 32, 1, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 32, 2, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 32, 2, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 32, 4, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 32, 3, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 32, 6, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 32, 7, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 32, 14, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 32, 15, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 32, 30, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 32, 31, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 32, 62, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 32, 63, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 32, 126, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 32, 127, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 32, 254, scalar_t>
                        <<<blocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 64) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 64, 1, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 64, 2, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 64, 2, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 64, 4, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 64, 3, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 64, 6, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 64, 7, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 64, 14, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 64, 15, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 64, 30, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 64, 31, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 64, 62, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 64, 63, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 64, 126, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 64, 127, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 64, 254, scalar_t>
                        <<<blocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 96) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 96, 1, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 96, 2, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 96, 2, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 96, 4, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 96, 3, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 96, 6, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 96, 7, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 96, 14, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 96, 15, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 96, 30, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 96, 31, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 96, 62, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 96, 63, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 96, 126, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 96, 127, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 96, 254, scalar_t>
                        <<<blocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 128) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 128, 1, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 128, 2, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 128, 2, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 128, 4, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 128, 3, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 128, 6, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 128, 7, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 128, 14, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 128, 15, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 128, 30, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 128, 31, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 128, 62, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 128, 63, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 128, 126, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 128, 127, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 128, 254, scalar_t>
                        <<<blocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 160) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 160, 1, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 160, 2, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 160, 2, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 160, 4, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 160, 3, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 160, 6, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 160, 7, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 160, 14, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 160, 15, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 160, 30, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 160, 31, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 160, 62, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 160, 63, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 160, 126, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 160, 127, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 160, 254, scalar_t>
                        <<<blocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 192) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 192, 1, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 192, 2, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 192, 2, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 192, 4, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 192, 3, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 192, 6, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 192, 7, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 192, 14, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 192, 15, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 192, 30, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 192, 31, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 192, 62, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 192, 63, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 192, 126, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 192, 127, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 192, 254, scalar_t>
                        <<<blocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 224) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 224, 1, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 224, 2, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 224, 2, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 224, 4, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 224, 3, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 224, 6, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 224, 7, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 224, 14, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 224, 15, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 224, 30, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 224, 31, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 224, 62, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 224, 63, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 224, 126, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 224, 127, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 224, 254, scalar_t>
                        <<<blocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 256) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 256, 1, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 256, 2, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 256, 2, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 256, 4, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 256, 3, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 256, 6, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 256, 7, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 256, 14, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 256, 15, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 256, 30, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 256, 31, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 256, 62, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 256, 63, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 256, 126, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 256, 127, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 256, 254, scalar_t>
                        <<<blocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 288) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 288, 1, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 288, 2, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 288, 2, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 288, 4, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 288, 3, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 288, 6, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 288, 7, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 288, 14, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 288, 15, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 288, 30, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 288, 31, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 288, 62, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 288, 63, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 288, 126, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 288, 127, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 288, 254, scalar_t>
                        <<<blocks, 288, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 320) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 320, 1, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 320, 2, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 320, 2, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 320, 4, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 320, 3, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 320, 6, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 320, 7, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 320, 14, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 320, 15, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 320, 30, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 320, 31, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 320, 62, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 320, 63, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 320, 126, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 320, 127, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 320, 254, scalar_t>
                        <<<blocks, 320, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 352) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 352, 1, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 352, 2, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 352, 2, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 352, 4, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 352, 3, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 352, 6, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 352, 7, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 352, 14, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 352, 15, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 352, 30, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 352, 31, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 352, 62, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 352, 63, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 352, 126, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 352, 127, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 352, 254, scalar_t>
                        <<<blocks, 352, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 384) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 384, 1, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 384, 2, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 384, 2, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 384, 4, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 384, 3, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 384, 6, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 384, 7, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 384, 14, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 384, 15, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 384, 30, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 384, 31, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 384, 62, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 384, 63, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 384, 126, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 384, 127, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 384, 254, scalar_t>
                        <<<blocks, 384, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 416) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 416, 1, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 416, 2, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 416, 2, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 416, 4, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 416, 3, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 416, 6, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 416, 7, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 416, 14, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 416, 15, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 416, 30, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 416, 31, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 416, 62, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 416, 63, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 416, 126, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 416, 127, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 416, 254, scalar_t>
                        <<<blocks, 416, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 448) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 448, 1, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 448, 2, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 448, 2, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 448, 4, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 448, 3, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 448, 6, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 448, 7, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 448, 14, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 448, 15, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 448, 30, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 448, 31, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 448, 62, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 448, 63, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 448, 126, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 448, 127, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 448, 254, scalar_t>
                        <<<blocks, 448, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 480) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 480, 1, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 480, 2, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 480, 2, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 480, 4, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 480, 3, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 480, 6, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 480, 7, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 480, 14, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 480, 15, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 480, 30, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 480, 31, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 480, 62, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 480, 63, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 480, 126, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 480, 127, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 480, 254, scalar_t>
                        <<<blocks, 480, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    if (sequenceLength <= 512) {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 512, 1, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 512, 2, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 512, 2, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 512, 4, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 512, 3, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 512, 6, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 512, 7, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 512, 14, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 512, 15, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 512, 30, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 512, 31, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 512, 62, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 512, 63, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 512, 126, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 512, 127, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 512, 254, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    } else

    {
        switch(filterSize) {

            case 3:

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 512, 1, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<3, 512, 2, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 5:

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 512, 2, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<5, 512, 4, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 7:

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 512, 3, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<7, 512, 6, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 15:

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 512, 7, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<15, 512, 14, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 31:

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 512, 15, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<31, 512, 30, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 63:

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 512, 31, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<63, 512, 62, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 127:

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 512, 63, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<127, 512, 126, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            case 255:

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 512, 127, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_forward", ([&] {
                        lightconv_forward_kernel<255, 512, 254, scalar_t>
                        <<<blocks, 512, 0, stream>>>(
                                input.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                output.data<scalar_t>());
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping forward pass" << std::endl;
                }
                break;

            default:
                std::cout << "WARNING: Unsupported filter length passed - skipping forward pass" << std::endl;
        }

    }

    return {output};
}
