#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <torch/csrc/autograd/grad_mode.h>
#include "HignnModel.hpp"
#include "Typedef.hpp"

#if USE_GPU
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#endif

using namespace std;

namespace {

std::size_t CheckedSizeProduct(
    const std::size_t a,
    const std::size_t b,
    const char *label) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error(
        std::string("CloseDot size overflow while calculating ") + label);
  }
  return a * b;
}

}  // namespace

void HignnModel::CloseDot(DeviceDoubleMatrix u, DeviceDoubleMatrix f, DeviceDoubleMatrix divM) {
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  // Captures the current time to start measuring elapsed time for performance
  // tracking.

  if (mMPIRank == 0)
    std::cout << "start of CloseDot" << std::endl;

  // Timing variables to track the execution duration of query and dot
  // operations.
  double queryDuration = 0;
  double dotDuration = 0;

  // Set the total number of close node pairs and the maximum size of the batch
  // that will be processed at once.
  const int closeNodeSize = mCloseMatIPtr->extent(0);
  // Stores the number of close node pairs to be processed.
  const int maxWorkSize = 1000;  // Maximum close node pairs per batch.
  int workSize = std::min(maxWorkSize, closeNodeSize);
  // Close node pairs for current batch, constrained by
  // maxWorkSize and the number of close node pairs.
  int finishedNodeSize = 0;  // Number of node pairs that have been processed.

  // Variables to track the total number of queries and iterations processed.
  std::size_t totalNumQuery = 0;
  std::size_t totalNumIter = 0;

  auto &mCloseMatI = *mCloseMatIPtr;
  auto &mCloseMatJ = *mCloseMatJPtr;
  auto &mCoord = *mCoordPtr;
  auto &mClusterTree = *mClusterTreePtr;

  bool useSymmetry = mUseSymmetry;
  const bool computeDivM = mComputeDivM;

  if (mCloseDotDebugFlag && mMPIRank == 0) {
    std::cout << "[CloseDot] compute_divM = " << computeDivM << std::endl;
  }

  {
    auto closeIHost =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mCloseMatI);

    auto closeJHost =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mCloseMatJ);

    int localOffDiag = 0;
    int localJGreater = 0;
    int localIGreater = 0;
    int localEqual = 0;

    for (int idx = 0; idx < closeNodeSize; idx++) {
      const int nodeI = closeIHost(idx);
      const int nodeJ = closeJHost(idx);

      if (nodeI == nodeJ) {
        localEqual++;
      } else {
        localOffDiag++;

        if (nodeJ > nodeI) {
          localJGreater++;
        } else {
          localIGreater++;
        }
      }
    }

    int globalOffDiag = 0;
    int globalJGreater = 0;
    int globalIGreater = 0;
    int globalEqual = 0;

    MPI_Allreduce(&localOffDiag, &globalOffDiag, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localJGreater, &globalJGreater, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localIGreater, &globalIGreater, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localEqual, &globalEqual, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (mMPIRank == 0) {
      std::cout << "[CloseDot close-pair orientation]\n"
                << "equal node pairs:      " << globalEqual << "\n"
                << "off-diagonal pairs:    " << globalOffDiag << "\n"
                << "nodeJ > nodeI pairs:   " << globalJGreater << "\n"
                << "nodeI > nodeJ pairs:   " << globalIGreater << "\n";
    }
  }

  {
    auto closeIHost =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mCloseMatI);
    auto closeJHost =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mCloseMatJ);
    auto clusterTreeHost =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), mClusterTree);

    const std::size_t numParticles =
        static_cast<std::size_t>(mCoord.extent(0));

    std::size_t totalCloseEntries = 0;
    std::size_t maxCloseEntriesPerBlock = 0;
    std::size_t minLeafSize = std::numeric_limits<std::size_t>::max();
    std::size_t maxLeafSize = 0;

    for (int idx = 0; idx < closeNodeSize; idx++) {
      const int nodeI = closeIHost(idx);
      const int nodeJ = closeJHost(idx);
      const std::size_t leafSizeI =
          static_cast<std::size_t>(
              clusterTreeHost(nodeI, 3) - clusterTreeHost(nodeI, 2));
      const std::size_t leafSizeJ =
          static_cast<std::size_t>(
              clusterTreeHost(nodeJ, 3) - clusterTreeHost(nodeJ, 2));
      const std::size_t entries =
          CheckedSizeProduct(
              leafSizeI,
              leafSizeJ,
              "close block entry count");

      if (entries >
          std::numeric_limits<std::size_t>::max() - totalCloseEntries) {
        throw std::overflow_error(
            "CloseDot size overflow while summing close entries");
      }
      totalCloseEntries += entries;
      maxCloseEntriesPerBlock =
          std::max(maxCloseEntriesPerBlock, entries);
      minLeafSize =
          std::min(minLeafSize, std::min(leafSizeI, leafSizeJ));
      maxLeafSize =
          std::max(maxLeafSize, std::max(leafSizeI, leafSizeJ));
    }

    if (closeNodeSize == 0) {
      minLeafSize = 0;
    }

    const std::size_t denseIndexStart =
        closeNodeSize == 1
            ? static_cast<std::size_t>(clusterTreeHost(closeIHost(0), 2))
            : 0;
    const std::size_t denseIndexEnd =
        closeNodeSize == 1
            ? static_cast<std::size_t>(clusterTreeHost(closeIHost(0), 3))
            : 0;
    const std::size_t denseParticleCount =
        denseIndexEnd - denseIndexStart;

    const bool denseAllPairsClose =
        closeNodeSize == 1 &&
        closeIHost(0) == closeJHost(0) &&
        denseParticleCount == numParticles;

    if (mMPIRank == 0) {
      std::cout << "[CloseDot debug]\n"
                << "N: " << numParticles << "\n"
                << "leaf size min/max: " << minLeafSize << " / "
                << maxLeafSize << "\n"
                << "number of close blocks: " << closeNodeSize << "\n"
                << "total close entries: " << totalCloseEntries << "\n"
                << "max close entries per block: "
                << maxCloseEntriesPerBlock << "\n"
                << "chunk size: " << mMaxRelativeCoord << "\n";
    }

    if (denseAllPairsClose) {
      const std::size_t rankTargetBegin =
          denseIndexStart +
          (denseParticleCount * static_cast<std::size_t>(mMPIRank)) /
          static_cast<std::size_t>(mMPISize);
      const std::size_t rankTargetEnd =
          denseIndexStart +
          (denseParticleCount * static_cast<std::size_t>(mMPIRank + 1)) /
          static_cast<std::size_t>(mMPISize);

      if (mMPIRank == 0) {
        std::cout
            << "[CloseDot dense streaming] using all-pairs CloseDot path"
            << std::endl;
      }
      std::cout << "[CloseDot dense streaming rank " << mMPIRank
                << "] target-row range: [" << rankTargetBegin << ", "
                << rankTargetEnd << ")" << std::endl;

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

      const std::size_t maxChunkEntries =
          std::max<std::size_t>(
              1,
              static_cast<std::size_t>(mMaxRelativeCoord));

      const std::size_t sourceChunkSize =
          std::max<std::size_t>(
              1,
              std::min(denseParticleCount, maxChunkEntries));

      const float minDivMRelativeDistance2 = 1e-12f;

      for (std::size_t targetBegin = rankTargetBegin;
           targetBegin < rankTargetEnd;
           targetBegin++) {
        for (std::size_t sourceBegin = denseIndexStart;
             sourceBegin < denseIndexEnd;
             sourceBegin += sourceChunkSize) {
          const std::size_t sourceEnd =
              std::min(sourceBegin + sourceChunkSize, denseIndexEnd);
          const std::size_t chunkEntries =
              sourceEnd - sourceBegin;

          CheckedSizeProduct(
              chunkEntries,
              static_cast<std::size_t>(3),
              "dense streaming relative-coordinate chunk");

          DeviceFloatVector relativeCoordChunk(
              "CloseDot_dense_relativeCoordChunk",
              chunkEntries * 3);

          Kokkos::parallel_for(
              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                  0,
                  chunkEntries),
              KOKKOS_LAMBDA(const std::size_t localSource) {
                const std::size_t sourceIndex =
                    sourceBegin + localSource;
                for (int d = 0; d < 3; d++) {
                  relativeCoordChunk(3 * localSource + d) =
                      mCoord(sourceIndex, d) -
                      mCoord(targetBegin, d);
                }
              });
          Kokkos::fence();

          std::chrono::steady_clock::time_point begin =
              std::chrono::steady_clock::now();

          torch::Tensor resultTensorContiguous;
          {
            torch::NoGradGuard noGrad;
            if (mCloseDotDebugFlag && mMPIRank == 0) {
              std::cout
                  << "[CloseDot dense streaming] torch grad enabled before "
                  << "velocity inference = "
                  << torch::GradMode::is_enabled()
                  << std::endl;
            }

            torch::Tensor relativeCoordTensor =
                torch::from_blob(
                    relativeCoordChunk.data(),
                    {static_cast<int64_t>(chunkEntries), 3},
                    options)
                    .clone()
                    .detach();

            std::vector<c10::IValue> inputs;
            inputs.push_back(relativeCoordTensor);

            auto rawTensor =
                mTwoBodyModel.forward(inputs)
                    .toTensor()
                    .view({static_cast<int64_t>(chunkEntries), 3, 3});
            auto mobilityTensor =
                0.5 * (rawTensor + rawTensor.transpose(1, 2));
            resultTensorContiguous =
                mobilityTensor
                    .contiguous()
                    .view({static_cast<int64_t>(chunkEntries), 9})
                    .contiguous();

            relativeCoordTensor = torch::Tensor();
            rawTensor = torch::Tensor();
            mobilityTensor = torch::Tensor();
          }

          std::chrono::steady_clock::time_point end =
              std::chrono::steady_clock::now();
          queryDuration +=
              std::chrono::duration_cast<std::chrono::microseconds>(
                  end - begin)
                  .count();

          begin = std::chrono::steady_clock::now();

          auto dataPtr =
              resultTensorContiguous.data_ptr<float>();

          Kokkos::parallel_for(
              Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                  0,
                  chunkEntries),
              KOKKOS_LAMBDA(const std::size_t localSource) {
                const std::size_t sourceIndex =
                    sourceBegin + localSource;
                for (int row = 0; row < 3; row++) {
                  double sum = 0.0;
                  for (int col = 0; col < 3; col++) {
                    sum +=
                        dataPtr[9 * localSource + row * 3 + col] *
                        f(sourceIndex, col);
                  }
                  Kokkos::atomic_add(
                      &u(targetBegin, row),
                      sum);
                }
              });
          Kokkos::fence();

          resultTensorContiguous = torch::Tensor();

          end = std::chrono::steady_clock::now();
          dotDuration +=
              std::chrono::duration_cast<std::chrono::microseconds>(
                  end - begin)
                  .count();

          if (computeDivM) {
#if USE_GPU
            size_t freeBytes = 0;
            size_t totalBytes = 0;
            cudaMemGetInfo(&freeBytes, &totalBytes);
            std::cout
                << "[CloseDot dense streaming rank " << mMPIRank
                << "] CUDA free before derivative chunk = "
                << freeBytes / 1024.0 / 1024.0 / 1024.0
                << " GiB" << std::endl;
#endif

            begin =
                std::chrono::steady_clock::now();

            DeviceFloatMatrix divMChunk(
                "CloseDot_dense_divMChunk",
                chunkEntries,
                3);
            auto hostDivMChunk =
                Kokkos::create_mirror_view(divMChunk);
            auto hostRelativeCoordChunk =
                Kokkos::create_mirror_view(relativeCoordChunk);
            Kokkos::deep_copy(hostRelativeCoordChunk, relativeCoordChunk);

            for (std::size_t localSource = 0;
                 localSource < chunkEntries;
                 localSource++) {
              for (int d = 0; d < 3; d++) {
                hostDivMChunk(localSource, d) = 0.0f;
              }
            }

            torch::Tensor relativeCoordTensor =
                torch::from_blob(
                    relativeCoordChunk.data(),
                    {static_cast<int64_t>(chunkEntries), 3},
                    options)
                    .clone()
                    .detach();
            relativeCoordTensor.set_requires_grad(true);

            std::vector<c10::IValue> inputs;
            inputs.push_back(relativeCoordTensor);

            {
              torch::AutoGradMode gradGuard(true);
              if (mCloseDotDebugFlag && mMPIRank == 0) {
                std::cout
                    << "[CloseDot dense streaming] torch grad enabled before "
                    << "divergence eval = "
                    << torch::GradMode::is_enabled()
                    << std::endl;
              }

              auto rawTensor =
                  mTwoBodyModel.forward(inputs)
                      .toTensor()
                      .view({static_cast<int64_t>(chunkEntries), 3, 3});
              auto mobilityTensor =
                  0.5 * (rawTensor + rawTensor.transpose(1, 2));

              for (int k = 0; k < 9; k++) {
                const int row = k / 3;
                const int col = k % 3;
                auto out =
                    mobilityTensor
                        .index({
                            torch::indexing::Slice(),
                            row,
                            col})
                        .sum();
                const bool retainGraph =
                    k < 8;
                auto grad =
                    torch::autograd::grad(
                        {out},
                        {relativeCoordTensor},
                        {},
                        retainGraph,
                        false,
                        false)[0]
                        .detach()
                        .to(torch::kCPU)
                        .contiguous();

                if (grad.numel() !=
                    static_cast<int64_t>(chunkEntries * 3)) {
                  throw std::runtime_error(
                      "Unexpected CloseDot dense streaming gradient size");
                }

                auto gradPtr =
                    grad.data_ptr<float>();

                for (std::size_t localSource = 0;
                     localSource < chunkEntries;
                     localSource++) {
                  const std::size_t sourceIndex =
                      sourceBegin + localSource;

                  const float dx =
                      hostRelativeCoordChunk(3 * localSource);
                  const float dy =
                      hostRelativeCoordChunk(3 * localSource + 1);
                  const float dz =
                      hostRelativeCoordChunk(3 * localSource + 2);
                  const float r2 =
                      dx * dx + dy * dy + dz * dz;

                  if (sourceIndex == targetBegin ||
                      !std::isfinite(r2) ||
                      r2 <= minDivMRelativeDistance2) {
                    continue;
                  }

                  const float gradDirect =
                      gradPtr[3 * localSource + col];

                  if (std::isfinite(gradDirect)) {
                    hostDivMChunk(localSource, row) += gradDirect;
                  }
                }

                grad = torch::Tensor();
              }

              rawTensor = torch::Tensor();
              mobilityTensor = torch::Tensor();
            }

            Kokkos::deep_copy(divMChunk, hostDivMChunk);

            Kokkos::parallel_for(
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    0,
                    chunkEntries),
                KOKKOS_LAMBDA(const std::size_t localSource) {
                  for (int row = 0; row < 3; row++) {
                    Kokkos::atomic_add(
                        &divM(targetBegin, row),
                        static_cast<double>(
                            divMChunk(localSource, row)));
                  }
                });
            Kokkos::fence();

            relativeCoordTensor = torch::Tensor();

#if USE_GPU
            c10::cuda::CUDACachingAllocator::emptyCache();
#endif

            end =
                std::chrono::steady_clock::now();
            queryDuration +=
                std::chrono::duration_cast<std::chrono::microseconds>(
                    end - begin)
                    .count();
          }

#if USE_GPU
          c10::cuda::CUDACachingAllocator::emptyCache();
#endif

          totalNumQuery += chunkEntries;
          totalNumIter++;
        }
      }

      MPI_Allreduce(MPI_IN_PLACE, &totalNumQuery, 1, MPI_UNSIGNED_LONG,
                    MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &totalNumIter, 1, MPI_UNSIGNED_LONG,
                    MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &queryDuration, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &dotDuration, 1, MPI_DOUBLE, MPI_MAX,
                    MPI_COMM_WORLD);

      if (mMPIRank == 0) {
        std::chrono::steady_clock::time_point t2 =
            std::chrono::steady_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
                .count();
        std::cout << "num query: " << totalNumQuery
                  << ", num iteration: " << totalNumIter
                  << ", query duration: " << queryDuration / 1e6
                  << "s, dot duration: " << dotDuration / 1e6 << "s"
                  << std::endl;
        printf("End of close dot. Dot time %.4fs\n",
               (double)duration / 1e6);
      }

      return;
    }
  }

  // Vectors for storing relative coordinates and node work assignments in the
  // legacy close-block batch path. The dense all-pairs path above uses its own
  // per-chunk scratch buffers and returns before these allocations.
  DeviceFloatVector relativeCoordPool("relativeCoordPool",
                                      CheckedSizeProduct(
                                          static_cast<std::size_t>(
                                              mMaxRelativeCoord),
                                          static_cast<std::size_t>(3),
                                          "legacy relative-coordinate pool"));

  DeviceIntVector workingNode("workingNode", maxWorkSize);

  DeviceIntVector relativeCoordSize("relativeCoordSize", maxWorkSize);
  DeviceIntVector relativeCoordOffset("relativeCoordOffset", maxWorkSize);

  // Begin processing node pairs in batches
  while (finishedNodeSize < closeNodeSize) {
    {
      workSize = min(maxWorkSize, closeNodeSize - finishedNodeSize);
      // Update work size based on remaining node pairs.

      // Define bounds for adjusting work size.
      int lowerWorkSize = 0;
      int upperWorkSize = workSize;

      // Dynamically adjust work size based on estimated workload.
      while (true) {
        int estimatedWorkload = 0;
        // Variable to store the estimated workload for the current batch.

        // Parallel reduction to estimate workload by summing the work sizes of
        // node pairs.
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
            KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
              const int nodeI = mCloseMatI(i + finishedNodeSize);
              const int indexIStart = mClusterTree(nodeI, 2);
              const int indexIEnd = mClusterTree(nodeI, 3);
              const int workSizeI = indexIEnd - indexIStart;

              const int nodeJ = mCloseMatJ(i + finishedNodeSize);
              const int indexJStart = mClusterTree(nodeJ, 2);
              const int indexJEnd = mClusterTree(nodeJ, 3);
              const int workSizeJ = indexJEnd - indexJStart;

              tSum += workSizeI * workSizeJ;  // Update the total estimated
                                              // workload for the batch.
            },
            Kokkos::Sum<int>(estimatedWorkload));

        // Adjustment of work size if estimated workload exceeds the maximum
        // allowed.
        if (estimatedWorkload > (int)mMaxRelativeCoord) {
          upperWorkSize = workSize;
          workSize = (lowerWorkSize + upperWorkSize) / 2;
          // Refine work size to distribute workload.
        } else {
          if (upperWorkSize - lowerWorkSize <= 1) {
            workSize = max(1, lowerWorkSize);
            // Finalize work size if difference between bounds is small.
            break;
          } else {
            lowerWorkSize = workSize;
            workSize = (lowerWorkSize + upperWorkSize) / 2;
            // Continue refining work size.
          }
        }
      }
    }

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i) {
          workingNode(i) = i + finishedNodeSize;
          // Assign node pairs to be processed in this batch.
        });
    Kokkos::fence();

    totalNumIter++;
    int totalCoord = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const std::size_t i, int &tSum) {
          const int rank = i;
          const int node = workingNode(rank);
          const int nodeI = mCloseMatI(node);
          const int nodeJ = mCloseMatJ(node);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          relativeCoordSize(rank) = workSizeI * workSizeJ;
          // Store the work size for this batch.

          tSum += workSizeI * workSizeJ;
          // Update the total coordinate count.
        },
        Kokkos::Sum<int>(totalCoord));
    Kokkos::fence();

    totalNumQuery += totalCoord;
    // Update the total number of queries.

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, workSize),
        KOKKOS_LAMBDA(const int rank) {
          relativeCoordOffset(rank) = 0;
          for (int i = 0; i < rank; i++) {
            relativeCoordOffset(rank) += relativeCoordSize(i);
            // Calculate offset for relative coordinates.
          }
        });
    Kokkos::fence();

    // Calculate the relative coordinates.
    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workSize,
                                                          Kokkos::AUTO()),
        KOKKOS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
                &teamMember) {
          const int rank = teamMember.league_rank();
          const int node = workingNode(rank);
          const int nodeI = mCloseMatI(node);
          const int nodeJ = mCloseMatJ(node);
          const int relativeOffset = relativeCoordOffset(rank);

          const int indexIStart = mClusterTree(nodeI, 2);
          const int indexIEnd = mClusterTree(nodeI, 3);
          const int indexJStart = mClusterTree(nodeJ, 2);
          const int indexJEnd = mClusterTree(nodeJ, 3);

          const int workSizeI = indexIEnd - indexIStart;
          const int workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
              [&](const int i) {
                int j = i / workSizeJ;
                int k = i % workSizeJ;

                const int index = relativeOffset + j * workSizeJ + k;
                for (int l = 0; l < 3; l++) {
                  relativeCoordPool(3 * index + l) =
                      mCoord(indexJStart + k, l) - mCoord(indexIStart + j, l);
                  // Calculate relative coordinate.
                }
              });
        });
    Kokkos::fence();

    // prepare the inference model.
// #if USE_GPU
//     auto options = torch::TensorOptions()
//                        .dtype(torch::kFloat32)
//                        .device(torch::kCUDA, mCudaDevice)
//                        .requires_grad(true); //change to true
// #else
//     auto options = torch::TensorOptions()
//                        .dtype(torch::kFloat32)
//                        .device(torch::kCPU)
//                        .requires_grad(true); //change to true
// #endif
#if USE_GPU
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCUDA, mCudaDevice);
#else
    auto options = torch::TensorOptions()
                       .dtype(torch::kFloat32)
                       .device(torch::kCPU);
#endif
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    DeviceFloatMatrix divMPairs("divMPairs", totalCoord, 3);
    DeviceFloatMatrix divMPairsT("divMPairsT", totalCoord, 3);

    Kokkos::parallel_for(
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, totalCoord * 3),
        KOKKOS_LAMBDA(const int i) {
          divMPairs(i / 3, i % 3) = 0.0;
          divMPairsT(i / 3, i % 3) = 0.0;
        });
    Kokkos::fence();

    torch::Tensor resultTensor_contiguous;

    if (computeDivM) {
      const float minDivMRelativeDistance2 = 1e-12f;
      DeviceIntVector validDivMPair("validDivMPair", totalCoord);
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, totalCoord),
          KOKKOS_LAMBDA(const int i) {
            const float dx = relativeCoordPool(3 * i);
            const float dy = relativeCoordPool(3 * i + 1);
            const float dz = relativeCoordPool(3 * i + 2);
            const float r2 = dx * dx + dy * dy + dz * dz;
            validDivMPair(i) =
                (isfinite(r2) && r2 > minDivMRelativeDistance2) ? 1 : 0;
          });
      Kokkos::fence();
      auto hostValidDivMPair = Kokkos::create_mirror_view(validDivMPair);
      Kokkos::deep_copy(hostValidDivMPair, validDivMPair);

      torch::Tensor relativeCoordTensor =
          torch::from_blob(relativeCoordPool.data(), {totalCoord, 3}, options)
              .clone()
              .detach();
      relativeCoordTensor.set_requires_grad(true);

      std::vector<c10::IValue> inputs;
      inputs.push_back(relativeCoordTensor);

      torch::AutoGradMode grad_guard(true);
      if (mCloseDotDebugFlag && mMPIRank == 0) {
        std::cout << "[CloseDot] torch grad enabled before divergence eval = "
                  << torch::GradMode::is_enabled() << std::endl;
      }
      auto rawTensor =
          mTwoBodyModel.forward(inputs).toTensor().view({totalCoord, 3, 3});
      auto mobilityTensor =
          0.5 * (rawTensor + rawTensor.transpose(1, 2));
      auto resultTensor =
          mobilityTensor.contiguous().view({totalCoord, 9});

      auto hostDivMPairs = Kokkos::create_mirror_view(divMPairs);
      auto hostDivMPairsT = Kokkos::create_mirror_view(divMPairsT);
      Kokkos::deep_copy(hostDivMPairs, divMPairs);
      Kokkos::deep_copy(hostDivMPairsT, divMPairsT);

      for (int k = 0; k < 9; k++) {
        const int row = k / 3;
        const int col = k % 3;
        auto out =
            mobilityTensor.index({
                torch::indexing::Slice(),
                row,
                col
            }).sum();
        const bool retainGraph = k < 8;
        auto grad = torch::autograd::grad({out}, {relativeCoordTensor}, {},
                                          retainGraph, false, false)[0]
                        .detach()
                        .to(torch::kCPU)
                        .contiguous();
        if (grad.numel() != totalCoord * 3) {
          throw std::runtime_error("Unexpected CloseDot gradient size");
        }
        auto gradPtr = grad.data_ptr<float>();

        for (int i = 0; i < totalCoord; i++) {
          if (!hostValidDivMPair(i)) {
            continue;
          }

          // Direct contribution:
          // div_i[row] += sum_col d M[row,col] / d r_col
          const float gradDirect = gradPtr[3 * i + col];

          if (std::isfinite(gradDirect)) {
            hostDivMPairs(i, row) += gradDirect;
          }

          // Transpose/symmetric reverse contribution:
          // div_j[col] += - sum_row d M[row,col] / d r_row
          //
          // The minus sign is applied later in the symmetry branch.
          const float gradTranspose = gradPtr[3 * i + row];

          if (std::isfinite(gradTranspose)) {
            hostDivMPairsT(i, col) += gradTranspose;
          }
        }
      }

      Kokkos::deep_copy(divMPairs, hostDivMPairs);
      Kokkos::deep_copy(divMPairsT, hostDivMPairsT);
      resultTensor_contiguous = resultTensor.detach().contiguous();
    } else {
      torch::Tensor relativeCoordTensor =
          torch::from_blob(relativeCoordPool.data(), {totalCoord, 3}, options)
              .clone()
              .detach();

      std::vector<c10::IValue> inputs;
      inputs.push_back(relativeCoordTensor);

      torch::AutoGradMode grad_guard(false);
      if (mCloseDotDebugFlag && mMPIRank == 0) {
        std::cout << "[CloseDot] torch grad enabled before velocity inference = "
                  << torch::GradMode::is_enabled() << std::endl;
      }
      auto rawTensor =
          mTwoBodyModel.forward(inputs).toTensor().view({totalCoord, 3, 3});
      auto mobilityTensor =
          0.5 * (rawTensor + rawTensor.transpose(1, 2));
      resultTensor_contiguous =
          mobilityTensor.contiguous().view({totalCoord, 9}).contiguous();
    }

    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    queryDuration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    begin = std::chrono::steady_clock::now();

    auto dataPtr = resultTensor_contiguous.data_ptr<float>();

    Kokkos::parallel_for(
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(workSize,
                                                          Kokkos::AUTO()),
        KOKKOS_LAMBDA(
            const Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type
                &teamMember) {
          const int rank = teamMember.league_rank();
          const int node = workingNode(rank);
          const int nodeI = mCloseMatI(node);
          const int nodeJ = mCloseMatJ(node);
          const int relativeOffset = relativeCoordOffset(rank);

          const std::size_t indexIStart = mClusterTree(nodeI, 2);
          const std::size_t indexIEnd = mClusterTree(nodeI, 3);
          const std::size_t indexJStart = mClusterTree(nodeJ, 2);
          const std::size_t indexJEnd = mClusterTree(nodeJ, 3);

          const std::size_t workSizeI = indexIEnd - indexIStart;
          const std::size_t workSizeJ = indexJEnd - indexJStart;

          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
              [&](const std::size_t index) {
                const std::size_t j = index / workSizeJ;
                const std::size_t k = index % workSizeJ;
                for (int row = 0; row < 3; row++) {
                  double sum = 0.0;
                  for (int col = 0; col < 3; col++)
                    sum +=
                        dataPtr[9 * (relativeOffset + index) + row * 3 + col] *
                        f(indexJStart + k, col);
                  Kokkos::atomic_add(&u(indexIStart + j, row), sum);
                  if (computeDivM) {
                    Kokkos::atomic_add(
                        &divM(indexIStart + j, row),
                        (double)divMPairs(relativeOffset + index, row));
                  }
                }
              });

          if (useSymmetry)
            if (nodeJ > nodeI) {
              Kokkos::parallel_for(
                  Kokkos::TeamThreadRange(teamMember, workSizeI * workSizeJ),
                  [&](const std::size_t index) {
                    const std::size_t j = index / workSizeJ;
                    const std::size_t k = index % workSizeJ;

                    for (int row = 0; row < 3; row++) {
                      double sum = 0.0;

                      // Symmetric velocity uses the transpose block:
                      //
                      // U_j[row] += sum_col M_ij[col,row] F_i[col]
                      for (int col = 0; col < 3; col++) {
                        sum +=
                            dataPtr[
                                9 * (relativeOffset + index) +
                                col * 3 + row] *
                            f(indexIStart + j, col);
                      }

                      Kokkos::atomic_add(
                          &u(indexJStart + k, row),
                          sum);

                      // Symmetric divergence:
                      //
                      // r = X_j - X_i
                      //
                      // Direct:
                      //   div_i[p] += sum_q d M[p,q] / d r_q
                      //
                      // Reverse transpose:
                      //   div_j[p] += - sum_q d M[q,p] / d r_q
                      //
                      // divMPairsT stores:
                      //   divMPairsT[p] = sum_q d M[q,p] / d r_q
                      if (computeDivM) {
                        Kokkos::atomic_add(
                            &divM(indexJStart + k, row),
                            -(double)divMPairsT(relativeOffset + index, row));
                      }
                    }
                  });
            }
        });
    Kokkos::fence();

    end = std::chrono::steady_clock::now();
    dotDuration +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - begin)
            .count();

    finishedNodeSize += workSize;
    // Update the count of processed node pairs.
  }

  MPI_Allreduce(MPI_IN_PLACE, &totalNumQuery, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  MPI_Allreduce(MPI_IN_PLACE, &totalNumIter, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  ;  // Aggregate total number of iterations across all MPI processes.
  MPI_Allreduce(MPI_IN_PLACE, &queryDuration, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  // Aggregate query duration across all MPI processes.
  MPI_Allreduce(MPI_IN_PLACE, &dotDuration, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  // Aggregate dot duration across all MPI processes.

  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
  // End the timer for performance tracking.

  if (mMPIRank == 0) {
    printf(
      "num query: %zu, num iteration: %zu, query duration: %.4fs, dot "
      "duration: %.4fs\n",
      totalNumQuery,
      totalNumIter,
      queryDuration / 1e6,
      dotDuration / 1e6);
    printf(
        "End of close dot. Dot time %.4fs\n",
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
            1e6);
  }
}
