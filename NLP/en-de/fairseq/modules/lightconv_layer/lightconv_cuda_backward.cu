
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lightconv_cuda.cuh"

std::vector<at::Tensor> lightconv_cuda_backward(
        at::Tensor gradOutput,
        int padding_l,
        at::Tensor input,
        at::Tensor filters) {

    // gradWrtInput
    const int minibatch = input.size(0);
    const int numFeatures = input.size(1);
    const int sequenceLength = input.size(2);

    const int numHeads = filters.size(0);
    const int filterSize = filters.size(1);

    const dim3 gradBlocks(minibatch, numFeatures);
    const dim3 weightGradFirstpassShortBlocks(minibatch, numHeads);
    const dim3 weightGradSecondpassBlocks(numHeads, filterSize);

    const int numFiltersInBlock = numFeatures / numHeads;

    auto gradInput = at::zeros_like(input);
    auto gradFilters = at::zeros_like(filters);

    at::DeviceGuard g(input.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    switch(filterSize) {

        case 3:

            if (sequenceLength <= 32) {

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<3, 32, 1, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<3, 32, 1, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<3, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<3, 32, 2, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<3, 32, 2, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<3, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 1) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<3, 32, 1, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<3, 32, 1, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<3, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<3, 32, 2, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<3, 32, 2, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<3, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 5:

            if (sequenceLength <= 32) {

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<5, 32, 2, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<5, 32, 2, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<5, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<5, 32, 4, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<5, 32, 4, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<5, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 2) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<5, 32, 2, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<5, 32, 2, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<5, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 4) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<5, 32, 4, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<5, 32, 4, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<5, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 7:

            if (sequenceLength <= 32) {

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<7, 32, 3, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<7, 32, 3, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<7, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<7, 32, 6, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<7, 32, 6, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<7, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 64) {

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<7, 64, 3, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<7, 64, 3, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<7, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<7, 64, 6, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<7, 64, 6, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<7, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 3) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<7, 32, 3, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<7, 32, 3, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<7, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 6) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<7, 32, 6, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<7, 32, 6, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<7, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 15:

            if (sequenceLength <= 32) {

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 32, 7, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 32, 7, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 32, 14, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 32, 14, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 64) {

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 64, 7, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 64, 7, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 64, 14, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 64, 14, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 96) {

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 96, 7, scalar_t>
                        <<<gradBlocks, 96, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 96, 7, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 96, scalar_t>
                        <<<weightGradSecondpassBlocks, 96, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 96, 14, scalar_t>
                        <<<gradBlocks, 96, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 96, 14, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 96, scalar_t>
                        <<<weightGradSecondpassBlocks, 96, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 128) {

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 128, 7, scalar_t>
                        <<<gradBlocks, 128, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 128, 7, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 128, scalar_t>
                        <<<weightGradSecondpassBlocks, 128, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 128, 14, scalar_t>
                        <<<gradBlocks, 128, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<15, 128, 14, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<15, 128, scalar_t>
                        <<<weightGradSecondpassBlocks, 128, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 7) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 32, 7, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<15, 32, 7, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<15, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 14) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<15, 32, 14, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<15, 32, 14, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<15, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 31:

            if (sequenceLength <= 32) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 32, 15, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 32, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 32, 30, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 32, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 64) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 64, 15, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 64, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 64, 30, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 64, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 96) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 96, 15, scalar_t>
                        <<<gradBlocks, 96, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 96, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 96, scalar_t>
                        <<<weightGradSecondpassBlocks, 96, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 96, 30, scalar_t>
                        <<<gradBlocks, 96, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 96, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 96, scalar_t>
                        <<<weightGradSecondpassBlocks, 96, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 128) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 128, 15, scalar_t>
                        <<<gradBlocks, 128, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 128, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 128, scalar_t>
                        <<<weightGradSecondpassBlocks, 128, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 128, 30, scalar_t>
                        <<<gradBlocks, 128, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 128, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 128, scalar_t>
                        <<<weightGradSecondpassBlocks, 128, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 160) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 160, 15, scalar_t>
                        <<<gradBlocks, 160, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 160, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 160, scalar_t>
                        <<<weightGradSecondpassBlocks, 160, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 160, 30, scalar_t>
                        <<<gradBlocks, 160, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 160, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 160, scalar_t>
                        <<<weightGradSecondpassBlocks, 160, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 192) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 192, 15, scalar_t>
                        <<<gradBlocks, 192, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 192, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 192, scalar_t>
                        <<<weightGradSecondpassBlocks, 192, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 192, 30, scalar_t>
                        <<<gradBlocks, 192, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 192, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 192, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 192, scalar_t>
                        <<<weightGradSecondpassBlocks, 192, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 224) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 224, 15, scalar_t>
                        <<<gradBlocks, 224, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 224, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 224, scalar_t>
                        <<<weightGradSecondpassBlocks, 224, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 224, 30, scalar_t>
                        <<<gradBlocks, 224, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 224, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 224, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 224, scalar_t>
                        <<<weightGradSecondpassBlocks, 224, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 256) {

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 256, 15, scalar_t>
                        <<<gradBlocks, 256, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 256, 15, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 256, scalar_t>
                        <<<weightGradSecondpassBlocks, 256, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 256, 30, scalar_t>
                        <<<gradBlocks, 256, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<31, 256, 30, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 256, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<31, 256, scalar_t>
                        <<<weightGradSecondpassBlocks, 256, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 15) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 32, 15, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<31, 32, 15, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<31, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 30) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<31, 32, 30, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<31, 32, 30, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<31, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 63:

            if (sequenceLength <= 32) {

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 32, 31, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 32, 31, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 32, 62, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 32, 62, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 64) {

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 64, 31, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 64, 31, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 64, 62, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 64, 62, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 96) {

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 96, 31, scalar_t>
                        <<<gradBlocks, 96, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 96, 31, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 96, scalar_t>
                        <<<weightGradSecondpassBlocks, 96, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 96, 62, scalar_t>
                        <<<gradBlocks, 96, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 96, 62, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 96, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 96, scalar_t>
                        <<<weightGradSecondpassBlocks, 96, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 128) {

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 128, 31, scalar_t>
                        <<<gradBlocks, 128, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 128, 31, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 128, scalar_t>
                        <<<weightGradSecondpassBlocks, 128, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 128, 62, scalar_t>
                        <<<gradBlocks, 128, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 128, 62, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 128, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 128, scalar_t>
                        <<<weightGradSecondpassBlocks, 128, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 160) {

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 160, 31, scalar_t>
                        <<<gradBlocks, 160, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 160, 31, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 160, scalar_t>
                        <<<weightGradSecondpassBlocks, 160, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 160, 62, scalar_t>
                        <<<gradBlocks, 160, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<63, 160, 62, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 160, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<63, 160, scalar_t>
                        <<<weightGradSecondpassBlocks, 160, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 31) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 32, 31, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<63, 32, 31, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<63, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 62) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<63, 32, 62, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<63, 32, 62, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<63, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 127:

            if (sequenceLength <= 32) {

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<127, 32, 63, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<127, 32, 63, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<127, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<127, 32, 126, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<127, 32, 126, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<127, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

            if (sequenceLength <= 64) {

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<127, 64, 63, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<127, 64, 63, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<127, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<127, 64, 126, scalar_t>
                        <<<gradBlocks, 64, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<127, 64, 126, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 64, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<127, 64, scalar_t>
                        <<<weightGradSecondpassBlocks, 64, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 63) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<127, 32, 63, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<127, 32, 63, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<127, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 126) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<127, 32, 126, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<127, 32, 126, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<127, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        case 255:

            if (sequenceLength <= 32) {

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<255, 32, 127, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<255, 32, 127, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<255, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<255, 32, 254, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numHeads, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_short_kernel<255, 32, 254, scalar_t>
                        <<<weightGradFirstpassShortBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                numHeads,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_short_kernel<255, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

            } else

                if (padding_l == 127) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<255, 32, 127, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<255, 32, 127, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<255, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                if (padding_l == 254) {
                    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "lightconv_backward", ([&] {
                        lightconv_grad_wrt_input_kernel<255, 32, 254, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                gradOutput.data<scalar_t>(),
                                filters.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                gradInput.data<scalar_t>());


                        at::Tensor tempSumGradFilters = at::zeros({minibatch, numFeatures, filterSize}, input.options().dtype(at::kFloat));
                        lightconv_grad_wrt_weights_firstpass_kernel<255, 32, 254, scalar_t>
                        <<<gradBlocks, 32, 0, stream>>>(
                                input.data<scalar_t>(),
                                gradOutput.data<scalar_t>(),
                                minibatch,
                                sequenceLength,
                                numFeatures,
                                numFiltersInBlock,
                                tempSumGradFilters.data<float>()
                        );

                        lightconv_grad_wrt_weights_secondpass_kernel<255, 32, scalar_t>
                        <<<weightGradSecondpassBlocks, 32, 0, stream>>>(
                                tempSumGradFilters.data<float>(),
                                minibatch,
                                numFiltersInBlock,
                                gradFilters.data<scalar_t>()
                        );
                    }));
                } else

                {
                    std::cout << "WARNING: Unsupported padding size - skipping backward pass" << std::endl;
                }

                break;

        default:
            std::cout << "WARNING: Unsupported filter length passed - skipping backward pass" << std::endl;

    }
    return {gradInput, gradFilters};
}
