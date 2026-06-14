#include "HignnModel.hpp"
#include <torch/csrc/autograd/grad_mode.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif

namespace {

torch::Tensor EvaluateDivBlocks(
    torch::Tensor relativeCoordTensor,
    torch::jit::Module &model) {

  relativeCoordTensor =
      relativeCoordTensor.clone().detach();

  relativeCoordTensor.set_requires_grad(true);

  std::vector<c10::IValue> inputs;
  inputs.push_back(relativeCoordTensor);

  torch::AutoGradMode enableGrad(true);

  auto raw =
      model.forward(inputs).toTensor().view({-1, 3, 3});

  auto mobility =
      0.5 * (raw + raw.transpose(1, 2));

  auto direct = torch::zeros(
      {mobility.size(0), 3},
      mobility.options());

  auto reverseTranspose = torch::zeros(
      {mobility.size(0), 3},
      mobility.options());

  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      auto output =
          mobility.index({
              torch::indexing::Slice(),
              row,
              col
          }).sum();

      const bool retainGraph =
          !(row == 2 && col == 2);

      auto grad =
          torch::autograd::grad(
              {output},
              {relativeCoordTensor},
              {},
              retainGraph,
              false,
              false
          )[0];

      // Direct contribution:
      //
      // direct[row] += d M[row,col] / d r_col
      direct.index_put_(
          {
              torch::indexing::Slice(),
              row
          },
          direct.index({
              torch::indexing::Slice(),
              row
          }) +
          grad.index({
              torch::indexing::Slice(),
              col
          })
      );

      // Reverse transpose contribution:
      //
      // reverseTranspose[col] += - d M[row,col] / d r_row
      //
      // This corresponds to:
      // div_j[p] = - sum_q d M[q,p] / d r_q
      reverseTranspose.index_put_(
          {
              torch::indexing::Slice(),
              col
          },
          reverseTranspose.index({
              torch::indexing::Slice(),
              col
          }) -
          grad.index({
              torch::indexing::Slice(),
              row
          })
      );
    }
  }

  // Output shape: [num_pairs, 6]
  //
  // columns 0,1,2: direct contracted divergence
  // columns 3,4,5: reverse transpose contracted divergence
  return torch::cat({direct, reverseTranspose}, 1)
      .detach()
      .contiguous();
}


torch::Tensor EvaluateContractedDivVectors(
    torch::Tensor relativeCoordTensor,
    torch::jit::Module &model) {

  return EvaluateDivBlocks(
      std::move(relativeCoordTensor),
      model);
}

}  // namespace

void HignnModel::FarDivDot(DeviceDoubleMatrix divM) {
  const auto startTime =
      std::chrono::steady_clock::now();

  if (mMPIRank == 0) {
    std::cout << "start of FarDivDot scalar ACA" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

//   if (mUseSymmetry) {
//     throw std::runtime_error(
//         "FarDivDot scalar ACA currently requires mUseSymmetry=false.");
//   }

  const int originalFarNodeSize =
      static_cast<int>(mFarMatIPtr->extent(0));

  if (originalFarNodeSize == 0) {
    return;
  }

  auto &mFarMatI = *mFarMatIPtr;
  auto &mFarMatJ = *mFarMatJPtr;
  auto &mClusterTree = *mClusterTreePtr;
  auto &mCoord = *mCoordPtr;


  // Effective directed far blocks.
  //
  // If mUseSymmetry == false:
  //   one effective block per stored far block.
  //
  // If mUseSymmetry == true:
  //   one direct effective block per stored far block,
  //   plus one reverse-transpose effective block for stored blocks with nodeJ > nodeI.
  // This matches the existing FarDot symmetry convention.
  DeviceIntVector effectiveOriginalBlock(
      "FarDivDot_effectiveOriginalBlock",
      std::max(1, 2 * originalFarNodeSize));

  DeviceIntVector effectiveIsReverse(
      "FarDivDot_effectiveIsReverse",
      std::max(1, 2 * originalFarNodeSize));

  int farNodeSize = 0;

  const int useSymmetry = mUseSymmetry ? 1 : 0;

  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          originalFarNodeSize),
      KOKKOS_LAMBDA(
          const int originalBlock,
          int &update,
          const bool final) {
      const int nodeI = mFarMatI(originalBlock);
      const int nodeJ = mFarMatJ(originalBlock);

      if (final) {
          effectiveOriginalBlock(update) = originalBlock;
          effectiveIsReverse(update) = 0;
      }

      update++;

      if (useSymmetry && nodeJ > nodeI) {
          if (final) {
            effectiveOriginalBlock(update) = originalBlock;
            effectiveIsReverse(update) = 1;
          }

        update++;
      }
    },
    farNodeSize);

  Kokkos::fence();

    // ============================================================
    // Debug: verify effective direct/reverse far-divergence blocks
    // ============================================================
    {
    int localDirectCount = 0;
    int localReverseCount = 0;

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, farNodeSize),
        KOKKOS_LAMBDA(const int i, int &count) {
            if (effectiveIsReverse(i) == 0) {
            count++;
            }
        },
        Kokkos::Sum<int>(localDirectCount));

    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, farNodeSize),
        KOKKOS_LAMBDA(const int i, int &count) {
            if (effectiveIsReverse(i) == 1) {
            count++;
            }
        },
        Kokkos::Sum<int>(localReverseCount));

    int globalDirectCount = 0;
    int globalReverseCount = 0;

    MPI_Allreduce(
        &localDirectCount,
        &globalDirectCount,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);

    MPI_Allreduce(
        &localReverseCount,
        &globalReverseCount,
        1,
        MPI_INT,
        MPI_SUM,
        MPI_COMM_WORLD);

    if (mMPIRank == 0) {
        std::cout
            << "[FarDivDot effective symmetry blocks]\n"
            << "stored far blocks:    " << originalFarNodeSize << "\n"
            << "effective far blocks: " << farNodeSize << "\n"
            << "direct blocks:        " << globalDirectCount << "\n"
            << "reverse blocks:       " << globalReverseCount << "\n";
    }
    }

  const int maximumIterations = mMaxIter;
  const double relativeTolerance = mEpsilon;
  constexpr double absolutePivotTolerance = 1.0e-12;
  constexpr double relativePivotTolerance = 1.0e-7;
  const double minimumPositiveDouble =
      std::numeric_limits<double>::min();

  DeviceIntVector validBlockFlag(
      "FarDivDot_validBlockFlag",
      farNodeSize);
  DeviceIntVector rowParticleCount(
      "FarDivDot_rowParticleCount",
      farNodeSize);
  DeviceIntVector columnCount(
      "FarDivDot_columnCount",
      farNodeSize);
  DeviceIntVector scalarRowCount(
      "FarDivDot_scalarRowCount",
      farNodeSize);
  DeviceIntVector fullScalarRank(
      "FarDivDot_fullScalarRank",
      farNodeSize);
  DeviceIntVector maximumBlockRank(
      "FarDivDot_maximumBlockRank",
      farNodeSize);
  DeviceIntVector acceptedRank(
      "FarDivDot_acceptedRank",
      farNodeSize);
  DeviceIntVector selectedColumn(
      "FarDivDot_selectedColumn",
      farNodeSize);
  DeviceIntVector selectedScalarRow(
      "FarDivDot_selectedScalarRow",
      farNodeSize);
  DeviceIntVector stopBlock(
      "FarDivDot_stopBlock",
      farNodeSize);
  DeviceDoubleVector approximationNormSquared(
      "FarDivDot_approximationNormSquared",
      farNodeSize);
  DeviceDoubleVector divergenceNormSquared(
      "FarDivDot_divergenceNormSquared",
      farNodeSize);
  DeviceIntVector consecutiveSmallDivUpdates(
      "FarDivDot_consecutiveSmallDivUpdates",
      farNodeSize);
  DeviceDoubleVector pivotValue(
      "FarDivDot_pivotValue",
      farNodeSize);
  DeviceDoubleVector columnScale(
      "FarDivDot_columnScale",
      farNodeSize);

  Kokkos::parallel_for(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(const int block) {
        const int originalBlock =
            effectiveOriginalBlock(block);

        const int reverse =
            effectiveIsReverse(block);

        const int originalNodeI =
            mFarMatI(originalBlock);

        const int originalNodeJ =
            mFarMatJ(originalBlock);

        // Effective row/target node and column/source node.
        const int nodeI =
            reverse ? originalNodeJ : originalNodeI;

        const int nodeJ =
            reverse ? originalNodeI : originalNodeJ;

        const int rows =
            mClusterTree(nodeI, 3) -
            mClusterTree(nodeI, 2);

        const int cols =
            mClusterTree(nodeJ, 3) -
            mClusterTree(nodeJ, 2);
        const int scalarRows = 3 * rows;
        const int fullRank =
            scalarRows < cols ? scalarRows : cols;
        const int positiveMaximumIterations =
            maximumIterations > 0 ? maximumIterations : 0;
        const int blockMaxRank =
            positiveMaximumIterations < fullRank
                ? positiveMaximumIterations
                : fullRank;

        rowParticleCount(block) = rows;
        columnCount(block) = cols;
        scalarRowCount(block) = scalarRows;
        fullScalarRank(block) = fullRank;
        maximumBlockRank(block) = blockMaxRank;
        acceptedRank(block) = 0;
        selectedColumn(block) = cols > 0 ? cols / 2 : 0;
        selectedScalarRow(block) = -1;
        stopBlock(block) = 0;
        approximationNormSquared(block) = 0.0;
        divergenceNormSquared(block) = 0.0;
        consecutiveSmallDivUpdates(block) = 0;
        pivotValue(block) = 0.0;
        columnScale(block) = 0.0;
        validBlockFlag(block) =
            (rows > 0 && cols > 0) ? 1 : 0;
      });
  Kokkos::fence();

  DeviceIntVector cFactorOffset(
      "FarDivDot_cFactorOffset",
      farNodeSize);
  DeviceIntVector qFactorOffset(
      "FarDivDot_qFactorOffset",
      farNodeSize);
  DeviceIntVector selectedHistoryOffset(
      "FarDivDot_selectedHistoryOffset",
      farNodeSize);

  int cFactorEntryCount = 0;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(
          const int block,
          int &update,
          const bool final) {
        if (final) {
          cFactorOffset(block) = update;
        }
        update +=
            scalarRowCount(block) *
            maximumBlockRank(block);
      },
      cFactorEntryCount);
  Kokkos::fence();

  int qFactorEntryCount = 0;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(
          const int block,
          int &update,
          const bool final) {
        if (final) {
          qFactorOffset(block) = update;
        }
        update +=
            columnCount(block) *
            maximumBlockRank(block);
      },
      qFactorEntryCount);
  Kokkos::fence();

  int selectedHistoryEntryCount = 0;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(
          const int block,
          int &update,
          const bool final) {
        if (final) {
          selectedHistoryOffset(block) = update;
        }
        update += maximumBlockRank(block);
      },
      selectedHistoryEntryCount);
  Kokkos::fence();

  DeviceDoubleVector cFactorPool(
    "FarDivDot_cFactorPool",
    std::max(1, cFactorEntryCount));
  DeviceDoubleVector qFactorPool(
    "FarDivDot_qFactorPool",
    std::max(1, qFactorEntryCount));

  // Stores the accumulated requested divergence contribution per scalar row:
  //
  //   blockDiv += C_k * sum_j Q_k[j]
  //
  // We reuse cFactorOffset(block) as the block offset because each valid block
  // has at least scalarRowCount(block) entries available in cFactorPool storage.
  DeviceDoubleVector blockDivergencePool(
    "FarDivDot_blockDivergencePool",
    std::max(1, cFactorEntryCount));

  DeviceIntVector selectedRowsHistory(
    "FarDivDot_selectedRowsHistory",
    std::max(1, selectedHistoryEntryCount));
  DeviceIntVector selectedColumnsHistory(
    "FarDivDot_selectedColumnsHistory",
    std::max(1, selectedHistoryEntryCount));

  Kokkos::deep_copy(cFactorPool, 0.0);
  Kokkos::deep_copy(qFactorPool, 0.0);
  Kokkos::deep_copy(blockDivergencePool, 0.0);
  Kokkos::deep_copy(selectedRowsHistory, -1);
  Kokkos::deep_copy(selectedColumnsHistory, -1);

  DeviceIntVector activeBlock(
      "FarDivDot_activeBlock",
      farNodeSize);
  DeviceIntVector rowActiveBlock(
      "FarDivDot_rowActiveBlock",
      farNodeSize);
  DeviceIntVector nextActiveBlock(
      "FarDivDot_nextActiveBlock",
      farNodeSize);
  DeviceIntVector relativeCoordOffset(
      "FarDivDot_relativeCoordOffset",
      farNodeSize);

  int activeBlockCount = 0;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(
          const int block,
          int &update,
          const bool final) {
        const bool active =
            validBlockFlag(block) != 0 &&
            maximumBlockRank(block) > 0;
        if (active) {
          if (final) {
            activeBlock(update) = block;
          }
          update++;
        }
      },
      activeBlockCount);
  Kokkos::fence();

  int initialColumnPairCapacity = 0;
  int initialRowPairCapacity = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(const int block, int &sum) {
        if (validBlockFlag(block) != 0 &&
            maximumBlockRank(block) > 0) {
          sum += rowParticleCount(block);
        }
      },
      Kokkos::Sum<int>(initialColumnPairCapacity));
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
          0,
          farNodeSize),
      KOKKOS_LAMBDA(const int block, int &sum) {
        if (validBlockFlag(block) != 0 &&
            maximumBlockRank(block) > 0) {
          sum += columnCount(block);
        }
      },
      Kokkos::Sum<int>(initialRowPairCapacity));
  Kokkos::fence();

  const int coordinatePoolPairCapacity =
      std::max(
          1,
          std::max(
              initialColumnPairCapacity,
              initialRowPairCapacity));

  DeviceFloatVector relativeCoordPool(
      "FarDivDot_relativeCoordPool",
      coordinatePoolPairCapacity * 3);

  auto evaluateContractedPoolToView =
      [&](const int pairCount,
          DeviceFloatMatrix contractedValues) -> int {
    if (pairCount <= 0) {
      return 0;
    }

#if USE_GPU
    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCUDA, mCudaDevice);
#else
    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(torch::kCPU);
#endif

    torch::Tensor relativeCoordTensor =
        torch::from_blob(
            relativeCoordPool.data(),
            {pairCount, 3},
            options);

    auto hostContractedValues =
        Kokkos::create_mirror_view(contractedValues);

    const int64_t chunkSize =
        std::max<int64_t>(
            1,
            static_cast<int64_t>(mMaxRelativeCoord));

    int chunkCount = 0;

    for (int64_t begin = 0;
         begin < static_cast<int64_t>(pairCount);
         begin += chunkSize) {
      const int64_t end =
          std::min<int64_t>(
              begin + chunkSize,
              static_cast<int64_t>(pairCount));

      auto relativeCoordChunk =
          relativeCoordTensor
              .slice(0, begin, end)
              .contiguous();

      std::cout
          << "[FarDivDot rank " << mMPIRank
          << "] derivative chunk rows = "
          << relativeCoordChunk.size(0)
          << std::endl;

#if USE_GPU
      size_t freeBytes = 0;
      size_t totalBytes = 0;
      cudaMemGetInfo(&freeBytes, &totalBytes);
      std::cout
          << "[FarDivDot rank " << mMPIRank
          << "] CUDA free before chunk = "
          << freeBytes / 1024.0 / 1024.0 / 1024.0
          << " GiB" << std::endl;
#endif

      auto contractedChunk =
          EvaluateContractedDivVectors(
              relativeCoordChunk,
              mTwoBodyModel);

      auto contractedChunkHost =
          contractedChunk
              .detach()
              .to(torch::kCPU)
              .contiguous();

      const float *contractedChunkPtr =
          contractedChunkHost.data_ptr<float>();

      const int64_t rows =
          end - begin;

      for (int64_t row = 0; row < rows; row++) {
        for (int component = 0; component < 6; component++) {
          hostContractedValues(begin + row, component) =
              contractedChunkPtr[6 * row + component];
        }
      }

      relativeCoordChunk = torch::Tensor();
      contractedChunk = torch::Tensor();
      contractedChunkHost = torch::Tensor();

#if USE_GPU
      c10::cuda::CUDACachingAllocator::emptyCache();
#endif

      chunkCount++;
    }

    Kokkos::deep_copy(contractedValues, hostContractedValues);

    return chunkCount;
  };

  long long localQueryCount = 0;
  long long localColumnTorchCallCount = 0;
  long long localRowTorchCallCount = 0;

  while (activeBlockCount > 0) {
    int columnPairCount = 0;
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            activeBlockCount),
        KOKKOS_LAMBDA(
            const int activeSlot,
            int &update,
            const bool final) {
          const int block =
              activeBlock(activeSlot);
          if (final) {
            relativeCoordOffset(activeSlot) = update;
          }
          update += rowParticleCount(block);
        },
        columnPairCount);
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(
            activeBlockCount,
            Kokkos::AUTO()),
        KOKKOS_LAMBDA(
            const Kokkos::TeamPolicy<
                Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
          const int activeSlot =
              teamMember.league_rank();
          const int block =
              activeBlock(activeSlot);
          const int originalBlock =
              effectiveOriginalBlock(block);

          const int reverse =
              effectiveIsReverse(block);

          const int originalNodeI =
              mFarMatI(originalBlock);

          const int originalNodeJ =
              mFarMatJ(originalBlock);

          // Effective target/source nodes.
          const int nodeI =
              reverse ? originalNodeJ : originalNodeI;

          const int nodeJ =
              reverse ? originalNodeI : originalNodeJ;

          const int indexIStart =
              mClusterTree(nodeI, 2);

          const int indexJ =
              mClusterTree(nodeJ, 2) +
              selectedColumn(block);
          const int rows =
              rowParticleCount(block);
          const int poolOffset =
              relativeCoordOffset(activeSlot);

          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(teamMember, rows),
              [&](const int localI) {
                const int poolIndex =
                    poolOffset + localI;
                for (int d = 0; d < 3; d++) {
                  if (reverse) {
                    // Reverse-transpose contribution still uses original r = X_J - X_I.
                    // Effective target is original J and effective source is original I,
                    // so r = target - source.
                    relativeCoordPool(3 * poolIndex + d) =
                        mCoord(indexIStart + localI, d) -
                        mCoord(indexJ, d);
                    } else {
                    // Direct contribution:
                    // r = source - target = X_J - X_I.
                    relativeCoordPool(3 * poolIndex + d) =
                        mCoord(indexJ, d) -
                        mCoord(indexIStart + localI, d);
                    }
                }
              });
        });
    Kokkos::fence();

    DeviceFloatMatrix contractedColumnValues(
        "FarDivDot_contractedColumnValues",
        columnPairCount,
        6);
    const int columnChunkCount =
        evaluateContractedPoolToView(
            columnPairCount,
            contractedColumnValues);
    localQueryCount += columnPairCount;
    localColumnTorchCallCount += columnChunkCount;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            activeBlockCount),
        KOKKOS_LAMBDA(const int activeSlot) {
          const int block =
              activeBlock(activeSlot);
          const int iter =
              acceptedRank(block);
          const int scalarRows =
              scalarRowCount(block);
          const int cols =
              columnCount(block);
          const int selectedCol =
              selectedColumn(block);
          const int queryOffset =
              relativeCoordOffset(activeSlot);
          const int cBase =
              cFactorOffset(block);
          const int qBase =
              qFactorOffset(block);
          const int cCurrent =
              cBase + iter * scalarRows;

          int invalidColumn = 0;
          int bestRow = -1;
          double bestMagnitude = -1.0;

          for (int scalarRow = 0;
               scalarRow < scalarRows;
               scalarRow++) {
            const int localI =
                scalarRow / 3;
            const int component =
                scalarRow % 3;

            const int componentOffset =
                effectiveIsReverse(block) ? 3 : 0;

            double value =
                static_cast<double>(
                    contractedColumnValues(
                        queryOffset + localI,
                        componentOffset + component));

            for (int previous = 0;
                 previous < iter;
                 previous++) {
              value -=
                  cFactorPool(
                      cBase +
                      previous * scalarRows +
                      scalarRow) *
                  qFactorPool(
                      qBase +
                      previous * cols +
                      selectedCol);
            }

            cFactorPool(cCurrent + scalarRow) = value;

            if (!isfinite(value)) {
              invalidColumn = 1;
              continue;
            }

            int rowAlreadySelected = 0;
            const int historyBase =
                selectedHistoryOffset(block);
            for (int previous = 0;
                 previous < iter;
                 previous++) {
              if (selectedRowsHistory(
                      historyBase + previous) ==
                  scalarRow) {
                rowAlreadySelected = 1;
                break;
              }
            }

            if (rowAlreadySelected == 0) {
              const double magnitude =
                  fabs(value);
              if (magnitude > bestMagnitude) {
                bestMagnitude = magnitude;
                bestRow = scalarRow;
              }
            }
          }

          selectedScalarRow(block) = bestRow;
          columnScale(block) = bestMagnitude;
          pivotValue(block) =
              bestRow >= 0
                  ? cFactorPool(cCurrent + bestRow)
                  : 0.0;

          const double pivotThreshold =
              absolutePivotTolerance +
              relativePivotTolerance *
                  (bestMagnitude > 0.0 ? bestMagnitude : 0.0);

          if (invalidColumn != 0 ||
              bestRow < 0 ||
              bestMagnitude < 0.0 ||
              !isfinite(pivotValue(block)) ||
              fabs(pivotValue(block)) <= pivotThreshold) {
            stopBlock(block) = 1;
          } else {
            stopBlock(block) = 0;
          }
        });
    Kokkos::fence();

    int rowActiveBlockCount = 0;
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            activeBlockCount),
        KOKKOS_LAMBDA(
            const int activeSlot,
            int &update,
            const bool final) {
          const int block =
              activeBlock(activeSlot);
          if (stopBlock(block) == 0) {
            if (final) {
              rowActiveBlock(update) = block;
            }
            update++;
          }
        },
        rowActiveBlockCount);
    Kokkos::fence();

    if (rowActiveBlockCount == 0) {
      break;
    }

    int rowPairCount = 0;
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            rowActiveBlockCount),
        KOKKOS_LAMBDA(
            const int activeSlot,
            int &update,
            const bool final) {
          const int block =
              rowActiveBlock(activeSlot);
          if (final) {
            relativeCoordOffset(activeSlot) = update;
          }
          update += columnCount(block);
        },
        rowPairCount);
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(
            rowActiveBlockCount,
            Kokkos::AUTO()),
        KOKKOS_LAMBDA(
            const Kokkos::TeamPolicy<
                Kokkos::DefaultExecutionSpace>::member_type &teamMember) {
          const int activeSlot =
              teamMember.league_rank();
          const int block =
              rowActiveBlock(activeSlot);
          const int originalBlock =
              effectiveOriginalBlock(block);

          const int reverse =
              effectiveIsReverse(block);

          const int originalNodeI =
              mFarMatI(originalBlock);

          const int originalNodeJ =
              mFarMatJ(originalBlock);

          // Effective target/source nodes.
          const int nodeI =
              reverse ? originalNodeJ : originalNodeI;

          const int nodeJ =
              reverse ? originalNodeI : originalNodeJ;

          const int indexIStart =
              mClusterTree(nodeI, 2);

          const int indexJStart =
              mClusterTree(nodeJ, 2);

          const int selectedLocalParticleI =
              selectedScalarRow(block) / 3;

          const int selectedGlobalI =
              indexIStart + selectedLocalParticleI;
          const int cols =
              columnCount(block);
          const int poolOffset =
              relativeCoordOffset(activeSlot);

          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(teamMember, cols),
              [&](const int localJ) {
                const int poolIndex =
                    poolOffset + localJ;
                for (int d = 0; d < 3; d++) {
                  if (reverse) {
                    // Reverse-transpose contribution uses original r = X_J - X_I.
                    // Effective target is original J and source is original I,
                    // so r = target - source.
                    relativeCoordPool(3 * poolIndex + d) =
                        mCoord(selectedGlobalI, d) -
                        mCoord(indexJStart + localJ, d);
                    } else {
                    // Direct contribution:
                    // r = source - target = X_J - X_I.
                    relativeCoordPool(3 * poolIndex + d) =
                        mCoord(indexJStart + localJ, d) -
                        mCoord(selectedGlobalI, d);
                    }
                }
              });
        });
    Kokkos::fence();

    DeviceFloatMatrix contractedRowValues(
        "FarDivDot_contractedRowValues",
        rowPairCount,
        6);
    const int rowChunkCount =
        evaluateContractedPoolToView(
            rowPairCount,
            contractedRowValues);
    localQueryCount += rowPairCount;
    localRowTorchCallCount += rowChunkCount;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            rowActiveBlockCount),
        KOKKOS_LAMBDA(const int activeSlot) {
          const int block =
              rowActiveBlock(activeSlot);
          const int iter =
              acceptedRank(block);
          const int scalarRows =
              scalarRowCount(block);
          const int cols =
              columnCount(block);
          const int selectedRow =
              selectedScalarRow(block);
          const int selectedComponent =
              selectedRow % 3;
          const int queryOffset =
              relativeCoordOffset(activeSlot);
          const int cBase =
              cFactorOffset(block);
          const int qBase =
              qFactorOffset(block);
          const int qCurrent =
              qBase + iter * cols;
          const double pivot =
              pivotValue(block);

          int invalidRow = 0;

          for (int localJ = 0;
               localJ < cols;
               localJ++) {
            const int componentOffset =
                effectiveIsReverse(block) ? 3 : 0;

            double value =
                static_cast<double>(
                    contractedRowValues(
                        queryOffset + localJ,
                        componentOffset + selectedComponent));

            for (int previous = 0;
                 previous < iter;
                 previous++) {
              value -=
                  cFactorPool(
                      cBase +
                      previous * scalarRows +
                      selectedRow) *
                  qFactorPool(
                      qBase +
                      previous * cols +
                      localJ);
            }

            value /= pivot;
            qFactorPool(qCurrent + localJ) = value;

            if (!isfinite(value)) {
              invalidRow = 1;
            }
          }

          if (invalidRow != 0) {
            stopBlock(block) = 1;
          }
        });
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            rowActiveBlockCount),
        KOKKOS_LAMBDA(const int activeSlot) {
          const int block =
              rowActiveBlock(activeSlot);

          if (stopBlock(block) != 0) {
            return;
          }

          const int iter =
              acceptedRank(block);
          const int originalBlock =
              effectiveOriginalBlock(block);

          const int reverse =
              effectiveIsReverse(block);

          const int originalNodeI =
              mFarMatI(originalBlock);

          const int originalNodeJ =
              mFarMatJ(originalBlock);

          // Effective target node.
          const int nodeI =
              reverse ? originalNodeJ : originalNodeI;

          const int indexIStart =
              mClusterTree(nodeI, 2);
          const int rows =
              rowParticleCount(block);
          const int scalarRows =
              scalarRowCount(block);
          const int cols =
              columnCount(block);
          const int selectedRow =
              selectedScalarRow(block);
          const int cBase =
              cFactorOffset(block);
          const int qBase =
              qFactorOffset(block);
          const int cCurrent =
              cBase + iter * scalarRows;
          const int qCurrent =
              qBase + iter * cols;

          double qSum = 0.0;

            for (int localJ = 0;
                localJ < cols;
                localJ++) {
            const double value =
                qFactorPool(qCurrent + localJ);
            qSum += value;
            }

            if (!isfinite(qSum)) {
            stopBlock(block) = 1;
            return;
            }

            // ------------------------------------------------------------
            // Requested-divergence update:
            //
            //   d_k = C_k * sum_j Q_k[j]
            //
            // Stop based on ||d_k|| / ||sum_l d_l||, not on ||C_k Q_k||.
            // ------------------------------------------------------------
            double updateNormSquared = 0.0;
            double newDivergenceNormSquared = 0.0;

            const int blockDivBase =
                cFactorOffset(block);

            int invalidContribution = 0;

            for (int scalarRow = 0;
                scalarRow < scalarRows;
                scalarRow++) {
            const double contribution =
                cFactorPool(cCurrent + scalarRow) *
                qSum;

            if (!isfinite(contribution)) {
                invalidContribution = 1;
                continue;
            }

            const double oldBlockValue =
                blockDivergencePool(blockDivBase + scalarRow);

            const double newBlockValue =
                oldBlockValue + contribution;

            blockDivergencePool(blockDivBase + scalarRow) =
                newBlockValue;

            updateNormSquared +=
                contribution * contribution;

            newDivergenceNormSquared +=
                newBlockValue * newBlockValue;
            }

            if (invalidContribution != 0 ||
                !isfinite(updateNormSquared) ||
                !isfinite(newDivergenceNormSquared)) {
            stopBlock(block) = 1;
            return;
            }

            divergenceNormSquared(block) =
                newDivergenceNormSquared;

            // Accumulate this accepted divergence update into the global divM.
            for (int localI = 0;
                localI < rows;
                localI++) {
            for (int component = 0;
                component < 3;
                component++) {
                const int scalarRow =
                    3 * localI + component;

                const double contribution =
                    cFactorPool(cCurrent + scalarRow) *
                    qSum;

                Kokkos::atomic_add(
                    &divM(indexIStart + localI, component),
                    contribution);
            }
            }

            const int historyBase =
                selectedHistoryOffset(block);
            selectedRowsHistory(historyBase + iter) =
                selectedRow;
            selectedColumnsHistory(historyBase + iter) =
                selectedColumn(block);

            acceptedRank(block) = iter + 1;

            // Keep this updated only as a diagnostic/legacy statistic.
            // It is no longer used for stopping.
            approximationNormSquared(block) =
                newDivergenceNormSquared;

            const double updateNorm =
                sqrt(
                    updateNormSquared > 0.0
                        ? updateNormSquared
                        : 0.0);

            const double divergenceNorm =
                sqrt(
                    newDivergenceNormSquared > minimumPositiveDouble
                        ? newDivergenceNormSquared
                        : minimumPositiveDouble);

            const bool divUpdateSmall =
                updateNorm <=
                relativeTolerance * divergenceNorm;

            // Require two consecutive small requested-divergence updates.
            // This avoids stopping too early from one cancellation in qSum.
            if (divUpdateSmall) {
            consecutiveSmallDivUpdates(block) += 1;
            } else {
            consecutiveSmallDivUpdates(block) = 0;
            }

            const bool toleranceReached =
                consecutiveSmallDivUpdates(block) >= 2;

            const bool rankLimitReached =
                iter + 1 >= maximumBlockRank(block) ||
                iter + 1 >= fullScalarRank(block);

            if (toleranceReached || rankLimitReached) {
            stopBlock(block) = 1;
            return;
            }

          int nextColumn = -1;
          double maximumRowResidual = 0.0;

          for (int localJ = 0;
               localJ < cols;
               localJ++) {
            int columnAlreadySelected = 0;
            for (int previous = 0;
                 previous <= iter;
                 previous++) {
              if (selectedColumnsHistory(
                      historyBase + previous) ==
                  localJ) {
                columnAlreadySelected = 1;
                break;
              }
            }

            if (columnAlreadySelected != 0) {
              continue;
            }

            const double candidate =
                fabs(qFactorPool(qCurrent + localJ));

            if (isfinite(candidate) &&
                candidate > maximumRowResidual) {
              maximumRowResidual = candidate;
              nextColumn = localJ;
            }
          }

          if (nextColumn < 0 ||
              maximumRowResidual <= absolutePivotTolerance) {
            stopBlock(block) = 1;
          } else {
            selectedColumn(block) = nextColumn;
            stopBlock(block) = 0;
          }
        });
    Kokkos::fence();

    int nextActiveBlockCount = 0;
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            rowActiveBlockCount),
        KOKKOS_LAMBDA(
            const int activeSlot,
            int &update,
            const bool final) {
          const int block =
              rowActiveBlock(activeSlot);
          if (stopBlock(block) == 0) {
            if (final) {
              nextActiveBlock(update) = block;
            }
            update++;
          }
        },
        nextActiveBlockCount);
    Kokkos::fence();

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,
            nextActiveBlockCount),
        KOKKOS_LAMBDA(const int activeSlot) {
          activeBlock(activeSlot) =
              nextActiveBlock(activeSlot);
        });
    Kokkos::fence();

    activeBlockCount = nextActiveBlockCount;
  }

  auto validBlockFlagHost =
      Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          validBlockFlag);
  auto fullScalarRankHost =
      Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          fullScalarRank);
  auto acceptedRankHost =
      Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(),
          acceptedRank);

  long long localFarBlockCount = 0;
  long long localAcceptedRankSum = 0;
  long long localFullRankSum = 0;
  long long localZeroRankBlocks = 0;
  long long localStoppedBeforeFullRank = 0;
  int localMinimumRank =
      std::numeric_limits<int>::max();
  int localMaximumRank = 0;
  double localPerBlockRankRatioSum = 0.0;

  for (int block = 0;
       block < farNodeSize;
       block++) {
    if (validBlockFlagHost(block) == 0) {
      continue;
    }

    const int rank =
        acceptedRankHost(block);
    const int fullRank =
        fullScalarRankHost(block);

    localFarBlockCount++;
    localAcceptedRankSum += rank;
    localFullRankSum += fullRank;

    if (rank == 0) {
      localZeroRankBlocks++;
    }

    if (rank > 0 && rank < fullRank) {
      localStoppedBeforeFullRank++;
    }

    localMinimumRank =
        std::min(localMinimumRank, rank);
    localMaximumRank =
        std::max(localMaximumRank, rank);

    if (fullRank > 0) {
      localPerBlockRankRatioSum +=
          static_cast<double>(rank) /
          static_cast<double>(fullRank);
    }
  }

  long long globalQueryCount = 0;
  long long globalColumnTorchCallCount = 0;
  long long globalRowTorchCallCount = 0;
  long long globalFarBlockCount = 0;
  long long globalAcceptedRankSum = 0;
  long long globalFullRankSum = 0;
  long long globalZeroRankBlocks = 0;
  long long globalStoppedBeforeFullRank = 0;
  double globalPerBlockRankRatioSum = 0.0;
  int globalMinimumRank = 0;
  int globalMaximumRank = 0;

  MPI_Allreduce(
      &localQueryCount,
      &globalQueryCount,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localColumnTorchCallCount,
      &globalColumnTorchCallCount,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localRowTorchCallCount,
      &globalRowTorchCallCount,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localFarBlockCount,
      &globalFarBlockCount,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localAcceptedRankSum,
      &globalAcceptedRankSum,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localFullRankSum,
      &globalFullRankSum,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localZeroRankBlocks,
      &globalZeroRankBlocks,
      1,
      MPI_LONG_LONG_INT,
      MPI_SUM,
      MPI_COMM_WORLD);
  MPI_Allreduce(
    &localStoppedBeforeFullRank,
    &globalStoppedBeforeFullRank,
    1,
    MPI_LONG_LONG_INT,
    MPI_SUM,
    MPI_COMM_WORLD);
  MPI_Allreduce(
      &localPerBlockRankRatioSum,
      &globalPerBlockRankRatioSum,
      1,
      MPI_DOUBLE,
      MPI_SUM,
      MPI_COMM_WORLD);

  const int localMinimumForReduction =
      localFarBlockCount > 0
          ? localMinimumRank
          : std::numeric_limits<int>::max();

  MPI_Allreduce(
      &localMinimumForReduction,
      &globalMinimumRank,
      1,
      MPI_INT,
      MPI_MIN,
      MPI_COMM_WORLD);
  MPI_Allreduce(
      &localMaximumRank,
      &globalMaximumRank,
      1,
      MPI_INT,
      MPI_MAX,
      MPI_COMM_WORLD);

  const auto endTime =
      std::chrono::steady_clock::now();

  if (mMPIRank == 0) {
    if (globalFarBlockCount == 0) {
      globalMinimumRank = 0;
    }

    const double averageRank =
        globalFarBlockCount > 0
            ? static_cast<double>(globalAcceptedRankSum) /
                  static_cast<double>(globalFarBlockCount)
            : 0.0;

    const double totalRankRatio =
        globalFullRankSum > 0
            ? static_cast<double>(globalAcceptedRankSum) /
                  static_cast<double>(globalFullRankSum)
            : 0.0;

    const double averagePerBlockRankRatio =
        globalFarBlockCount > 0
            ? globalPerBlockRankRatioSum /
                  static_cast<double>(globalFarBlockCount)
            : 0.0;

    const double elapsedSeconds =
        std::chrono::duration_cast<std::chrono::microseconds>(
            endTime - startTime)
            .count() /
        1.0e6;

    std::cout
        << "\n[FarDivDot contracted scalar ACA statistics]\n"
        << "Far blocks:                         "
        << globalFarBlockCount << "\n"
        << "Derivative queries:                 "
        << globalQueryCount << "\n"
        << "Batched column Torch calls:         "
        << globalColumnTorchCallCount << "\n"
        << "Batched row Torch calls:            "
        << globalRowTorchCallCount << "\n"
        << "Zero-rank blocks:                   "
        << globalZeroRankBlocks << "\n"
        << "Stopped before full rank:           "
        << globalStoppedBeforeFullRank << "\n"
        << "Total scalar ACA rank:              "
        << globalAcceptedRankSum << "\n"
        << "Average scalar ACA rank/block:      "
        << averageRank << "\n"
        << "Minimum scalar ACA rank:            "
        << globalMinimumRank << "\n"
        << "Maximum scalar ACA rank:            "
        << globalMaximumRank << "\n"
        << "Total scalar full rank:             "
        << globalFullRankSum << "\n"
        << "Total ACA rank / full rank (%):     "
        << 100.0 * totalRankRatio << "%\n"
        << "Average per-block rank ratio (%):   "
        << 100.0 * averagePerBlockRankRatio << "%\n"
        << "Elapsed time:                       "
        << elapsedSeconds << " s\n"
        << std::endl;
  }
}
