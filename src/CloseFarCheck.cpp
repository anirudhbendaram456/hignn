#include "HignnModel.hpp"
#include <limits>
#include <stdexcept>
#include <string>

namespace {

std::size_t CheckedSizeProduct(
    const std::size_t a,
    const std::size_t b,
    const char *label) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::overflow_error(
        std::string("[CloseFarCheck] overflow in ") + label);
  }
  return a * b;
}

std::size_t CheckedSizeSum(
    const std::size_t a,
    const std::size_t b,
    const char *label) {
  if (b > std::numeric_limits<std::size_t>::max() - a) {
    throw std::overflow_error(
        std::string("[CloseFarCheck] overflow in ") + label);
  }
  return a + b;
}

int ClampSizeToInt(const std::size_t value) {
  if (value >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return static_cast<int>(value);
}

}  // namespace

void HignnModel::CloseFarCheck() {
  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();

  if (mMPIRank == 0)
    std::cout << "start of CloseFarCheck" << std::endl;

  mLeafNodeList.clear();

  auto &mClusterTree = *mClusterTreeMirrorPtr;

  const std::size_t numParticles = mCoordPtr->extent(0);

  if (numParticles <= 1) {
    if (mMPIRank == 0) {
      std::cout << "CloseFarCheck: N <= 1, creating diagnostic self close-pair (0,0)."
                << std::endl;
    }

    mLeafNodeList.clear();

    if (mClusterTree.extent(0) > 0) {
      mLeafNodeList.push_back(0);
    }

    // One particle-particle block: node 0 with itself.
    mMaxCloseDotBlockSize = 1;

    // Create one close interaction pair: (0, 0)
    mCloseMatIPtr = std::make_shared<DeviceIndexVector>("mCloseMatI", 1);
    mCloseMatJPtr = std::make_shared<DeviceIndexVector>("mCloseMatJ", 1);

    auto closeIHost = Kokkos::create_mirror_view(*mCloseMatIPtr);
    auto closeJHost = Kokkos::create_mirror_view(*mCloseMatJPtr);

    closeIHost(0) = 0;
    closeJHost(0) = 0;

    Kokkos::deep_copy(*mCloseMatIPtr, closeIHost);
    Kokkos::deep_copy(*mCloseMatJPtr, closeJHost);

    // No far interactions for one particle.
    mFarMatIPtr = std::make_shared<DeviceIndexVector>("mFarMatI", 0);
    mFarMatJPtr = std::make_shared<DeviceIndexVector>("mFarMatJ", 0);

    MPI_Barrier(MPI_COMM_WORLD);
    return;
  }

  const std::size_t numTreeNodes = mClusterTree.extent(0);

  const long long rootChild0 =
      static_cast<long long>(mClusterTree(0, 0));
  const long long rootChild1 =
      static_cast<long long>(mClusterTree(0, 1));
  const long long rootBeginLL =
      static_cast<long long>(mClusterTree(0, 2));
  const long long rootEndLL =
      static_cast<long long>(mClusterTree(0, 3));

  const bool rootRangeValid =
      rootBeginLL >= 0 &&
      rootEndLL >= rootBeginLL;

  const std::size_t rootBegin =
      rootRangeValid ? static_cast<std::size_t>(rootBeginLL) : 0;
  const std::size_t rootEnd =
      rootRangeValid ? static_cast<std::size_t>(rootEndLL) : 0;
  const std::size_t rootSize =
      rootRangeValid ? (rootEnd - rootBegin) : 0;

  const bool rootCoversAllParticles =
      rootRangeValid &&
      rootBegin == 0 &&
      rootSize == numParticles;

  // In this codebase, child0 == 0 means leaf.
  const bool rootMarkedLeaf =
      rootChild0 == 0;

  // Your failing case:
  //   extent(0) = 3
  //   child0 = 3
  //   child1 = 4
  // Valid node indices are only 0,1,2, so these children are invalid.
  const bool rootChildrenOutOfRange =
      rootChild0 < 0 ||
      rootChild1 < 0 ||
      static_cast<std::size_t>(rootChild0) >= numTreeNodes ||
      static_cast<std::size_t>(rootChild1) >= numTreeNodes;

  const bool rootHasNoUsableChildren =
      rootMarkedLeaf || rootChildrenOutOfRange;

  const bool rootLeafCoversAllParticles =
      rootCoversAllParticles && rootHasNoUsableChildren;

  if (mMPIRank == 0) {
    std::cout << "[CloseFarCheck early debug]" << std::endl;
    std::cout << "  numParticles = " << numParticles << std::endl;
    std::cout << "  mClusterTree.extent(0) = " << numTreeNodes << std::endl;
    std::cout << "  root child0 = " << rootChild0 << std::endl;
    std::cout << "  root child1 = " << rootChild1 << std::endl;
    std::cout << "  root begin  = " << rootBegin << std::endl;
    std::cout << "  root end    = " << rootEnd << std::endl;
    std::cout << "  root size   = " << rootSize << std::endl;
    std::cout << "  rootCoversAllParticles = "
              << rootCoversAllParticles << std::endl;
    std::cout << "  rootMarkedLeaf = "
              << rootMarkedLeaf << std::endl;
    std::cout << "  rootChildrenOutOfRange = "
              << rootChildrenOutOfRange << std::endl;
    std::cout << "  rootLeafCoversAllParticles = "
              << rootLeafCoversAllParticles << std::endl;
  }

  if (rootLeafCoversAllParticles) {
    mLeafNodeList.clear();
    mLeafNodeList.push_back(0);

    mCloseMatIPtr = std::make_shared<DeviceIndexVector>("mCloseMatI", 1);
    mCloseMatJPtr = std::make_shared<DeviceIndexVector>("mCloseMatJ", 1);

    auto closeIHost = Kokkos::create_mirror_view(*mCloseMatIPtr);
    auto closeJHost = Kokkos::create_mirror_view(*mCloseMatJPtr);

    closeIHost(0) = 0;
    closeJHost(0) = 0;

    Kokkos::deep_copy(*mCloseMatIPtr, closeIHost);
    Kokkos::deep_copy(*mCloseMatJPtr, closeJHost);

    mFarMatIPtr = std::make_shared<DeviceIndexVector>("mFarMatI", 0);
    mFarMatJPtr = std::make_shared<DeviceIndexVector>("mFarMatJ", 0);

    mMaxCloseDotBlockSize = ClampSizeToInt(numParticles);

    const std::size_t totalCloseEntry =
        CheckedSizeProduct(
            numParticles,
            numParticles,
            "single-root-leaf close entry count");

    MPI_Barrier(MPI_COMM_WORLD);

    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
            .count();

    if (mMPIRank == 0) {
      std::cout << "[CloseFarCheck] single-root-leaf all-CloseDot fast path"
                << std::endl;
      std::cout << "  numParticles = " << numParticles << std::endl;
      std::cout << "  numTreeNodes = " << numTreeNodes << std::endl;
      std::cout << "Total close pair: 1" << std::endl;
      std::cout << "Total close entry: " << totalCloseEntry << std::endl;
      std::cout << "Admissible blocks (far pairs): 0" << std::endl;
      std::cout << "Inadmissible blocks (close pairs): 1" << std::endl;
      std::cout << "Total blocks: 1" << std::endl;
      std::cout << "Total far node: 0" << std::endl;
      std::cout << "Total far pair: 0" << std::endl;
      std::cout << "Total far entry: 0" << std::endl;
      std::cout << "Max single node size: " << numParticles << std::endl;
      std::cout << "Time for building close and far matrix: "
                << (double)duration / 1e6 << "s" << std::endl;
    }

    return;
  }

  auto hasValidChildren = [&](const std::size_t node) -> bool {
    const std::size_t nNodes = mClusterTree.extent(0);

    if (node >= nNodes) {
      return false;
    }

    const long long child0 =
        static_cast<long long>(mClusterTree(node, 0));
    const long long child1 =
        static_cast<long long>(mClusterTree(node, 1));

    // child0 == 0 is the leaf sentinel in this code.
    if (child0 == 0) {
      return false;
    }

    if (child0 < 0 || child1 < 0) {
      return false;
    }

    if (static_cast<std::size_t>(child0) >= nNodes ||
        static_cast<std::size_t>(child1) >= nNodes) {
      return false;
    }

    return true;
  };

  HostFloatMatrix mAux;
  Kokkos::resize(mAux, mClusterTree.extent(0), 6);

  // init mAux
  for (size_t i = 0; i < mAux.extent(0); i++)
    mAux(i, 0) = -std::numeric_limits<float>::max();

  int initialLevel = 0;
  while ((static_cast<std::size_t>(1) << (initialLevel + 1)) <=
         static_cast<std::size_t>(mMPISize)) {
    initialLevel++;
  }

  while (initialLevel > 0 &&
         ((static_cast<std::size_t>(1) << initialLevel) - 1) >=
             mClusterTree.extent(0)) {
    initialLevel--;
  }

  const std::size_t firstNodeAtInitialLevel =
      (static_cast<std::size_t>(1) << initialLevel) - 1;
  const std::size_t maxWorkSize =
      std::min<std::size_t>(
          static_cast<std::size_t>(mMPISize),
          static_cast<std::size_t>(1) << initialLevel);

  // go over all nodes, calculate aux based on the tree structure.
  // parallel stage
  if (static_cast<std::size_t>(mMPIRank) < maxWorkSize) {
    std::stack<std::size_t> workStack;
    std::vector<std::size_t> computeAuxStack;

    const std::size_t startNode =
        firstNodeAtInitialLevel +
        static_cast<std::size_t>(mMPIRank);

    if (startNode < mClusterTree.extent(0)) {
      workStack.push(startNode);
    }

    while (workStack.size() != 0) {
      auto node = workStack.top();

      workStack.pop();

      if (hasValidChildren(node)) {
        workStack.push(static_cast<std::size_t>(mClusterTree(node, 0)));
        workStack.push(static_cast<std::size_t>(mClusterTree(node, 1)));
      } else {
        computeAuxStack.push_back(node);
      }
    }

#pragma omp parallel for schedule(dynamic, 10)
    for (size_t i = 0; i < computeAuxStack.size(); i++) {
      auto node = computeAuxStack[i];

      std::vector<float> aux(6);
      ComputeAux(mClusterTree(node, 2), mClusterTree(node, 3), aux);

      mAux(node, 0) = aux[0];
      mAux(node, 1) = aux[1];
      mAux(node, 2) = aux[2];
      mAux(node, 3) = aux[3];
      mAux(node, 4) = aux[4];
      mAux(node, 5) = aux[5];
    }

    if (startNode < mClusterTree.extent(0)) {
      workStack.push(startNode);
    }
    while (workStack.size() != 0) {
      auto node = workStack.top();
      if (hasValidChildren(node)) {
        const std::size_t child0 =
            static_cast<std::size_t>(mClusterTree(node, 0));
        const std::size_t child1 =
            static_cast<std::size_t>(mClusterTree(node, 1));

        // check if child nodes have calculated aux.
        bool canContinue = false;

        if (mAux(child0, 0) == -std::numeric_limits<float>::max()) {
          workStack.push(child0);
          canContinue = true;
        }

        if (mAux(child1, 0) == -std::numeric_limits<float>::max()) {
          workStack.push(child1);
          canContinue = true;
        }

        if (!canContinue) {
          mAux(node, 0) = std::min(mAux(child0, 0), mAux(child1, 0));
          mAux(node, 1) = std::max(mAux(child0, 1), mAux(child1, 1));
          mAux(node, 2) = std::min(mAux(child0, 2), mAux(child1, 2));
          mAux(node, 3) = std::max(mAux(child0, 3), mAux(child1, 3));
          mAux(node, 4) = std::min(mAux(child0, 4), mAux(child1, 4));
          mAux(node, 5) = std::max(mAux(child0, 5), mAux(child1, 5));

          workStack.pop();
        }
      } else {
        workStack.pop();
      }
    }
  }

  for (std::size_t rank = 0; rank < maxWorkSize; rank++) {
    const std::size_t reorderedNode =
        firstNodeAtInitialLevel + rank;
    if (reorderedNode >= mClusterTree.extent(0)) {
      continue;
    }
    const size_t nodeStart = mClusterTree(reorderedNode, 0);
    const size_t nodeEnd =
        (rank == maxWorkSize - 1 ||
         reorderedNode + 1 >= mClusterTree.extent(0))
            ? mClusterTree.extent(0)
            : mClusterTree(reorderedNode + 1, 0);

    MPI_Bcast(mAux.data() + 6 * nodeStart, 6 * (nodeEnd - nodeStart), MPI_FLOAT,
              static_cast<int>(rank), MPI_COMM_WORLD);
  }

  // sequential stage
  {
    std::stack<std::size_t> workStack;
    workStack.push(0);
    while (workStack.size() != 0) {
      auto node = workStack.top();
      if (hasValidChildren(node)) {
        const std::size_t child0 =
            static_cast<std::size_t>(mClusterTree(node, 0));
        const std::size_t child1 =
            static_cast<std::size_t>(mClusterTree(node, 1));

        // check if child nodes have calculated aux.
        bool canContinue = false;

        if (mAux(child0, 0) == -std::numeric_limits<float>::max()) {
          workStack.push(child0);
          canContinue = true;
        }

        if (mAux(child1, 0) == -std::numeric_limits<float>::max()) {
          workStack.push(child1);
          canContinue = true;
        }

        if (!canContinue) {
          mAux(node, 0) = std::min(mAux(child0, 0), mAux(child1, 0));
          mAux(node, 1) = std::max(mAux(child0, 1), mAux(child1, 1));
          mAux(node, 2) = std::min(mAux(child0, 2), mAux(child1, 2));
          mAux(node, 3) = std::max(mAux(child0, 3), mAux(child1, 3));
          mAux(node, 4) = std::min(mAux(child0, 4), mAux(child1, 4));
          mAux(node, 5) = std::max(mAux(child0, 5), mAux(child1, 5));

          workStack.pop();
        }
      } else {
        std::vector<float> aux(6);
        ComputeAux(mClusterTree(node, 2), mClusterTree(node, 3), aux);

        mAux(node, 0) = aux[0];
        mAux(node, 1) = aux[1];
        mAux(node, 2) = aux[2];
        mAux(node, 3) = aux[3];
        mAux(node, 4) = aux[4];
        mAux(node, 5) = aux[5];

        workStack.pop();
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // estimate far and close pair size based on aux.
  // close far check
  std::vector<std::vector<int>> farMat(mClusterTree.extent(0));
  std::vector<std::vector<int>> closeMat(mClusterTree.extent(0));

  std::size_t totalEntry = 0;

  closeMat[0].push_back(0);
  std::queue<std::size_t> nodeList;
  nodeList.emplace(0);
  while (nodeList.size() != 0) {
    auto node = nodeList.front();
    auto i = node;
    nodeList.pop();

    if (mClusterTree(node, 0) != 0) {
      std::vector<int> childCloseMat;

      for (size_t j = 0; j < closeMat[i].size(); j++) {
        bool isFar = CloseFarCheck(mAux, node, closeMat[i][j]);

        std::size_t nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
        std::size_t nodeSizeJ =
            mClusterTree(closeMat[i][j], 3) - mClusterTree(closeMat[i][j], 2);

        isFar = isFar && nodeSizeI < mMaxRelativeCoord &&
                nodeSizeJ < mMaxRelativeCoord;

        if (isFar) {
          farMat[i].push_back(closeMat[i][j]);

          totalEntry =
              CheckedSizeSum(
                  totalEntry,
                  CheckedSizeProduct(
                      nodeSizeI,
                      nodeSizeJ,
                      "far candidate entry count"),
                  "total far candidate entry count");
        } else {
          if (mClusterTree(closeMat[i][j], 0) != 0) {
            childCloseMat.push_back(mClusterTree(closeMat[i][j], 0));
            childCloseMat.push_back(mClusterTree(closeMat[i][j], 1));
          } else {
            childCloseMat.push_back(closeMat[i][j]);
          }
        }
      }
      closeMat[mClusterTree(node, 0)] = childCloseMat;
      closeMat[mClusterTree(node, 1)] = childCloseMat;

      nodeList.emplace(mClusterTree(node, 0));
      nodeList.emplace(mClusterTree(node, 1));
    } else {
      std::vector<int> newCloseMat;

      for (size_t j = 0; j < closeMat[i].size(); j++) {
        bool isFar = CloseFarCheck(mAux, node, closeMat[i][j]);

        std::size_t nodeSizeI = mClusterTree(node, 3) - mClusterTree(node, 2);
        std::size_t nodeSizeJ =
            mClusterTree(closeMat[i][j], 3) - mClusterTree(closeMat[i][j], 2);

        totalEntry =
            CheckedSizeSum(
                totalEntry,
                CheckedSizeProduct(
                    nodeSizeI,
                    nodeSizeJ,
                    "close/far traversal entry count"),
                "total traversal entry count");

        if (isFar) {
          // need to make sure the column node is small enough
          if (nodeSizeJ < mMaxRelativeCoord)
            farMat[i].push_back(closeMat[i][j]);
          else {
            std::stack<int> workChildStack;
            workChildStack.push(closeMat[i][j]);

            while (workChildStack.size() != 0) {
              auto childNode = workChildStack.top();
              workChildStack.pop();
              size_t nodeSize =
                  mClusterTree(childNode, 3) - mClusterTree(childNode, 2);

              if (nodeSize < mMaxRelativeCoord) {
                farMat[i].push_back(childNode);
              } else {
                workChildStack.push(mClusterTree(childNode, 0));
                workChildStack.push(mClusterTree(childNode, 1));
              }
            }
          }
        } else {
          // need to make sure this is a leaf node for close mat.
          if (mClusterTree(closeMat[i][j], 0) == 0)
            newCloseMat.push_back(closeMat[i][j]);
          else {
            std::vector<int> childCloseMat;
            std::stack<int> workChildStack;
            workChildStack.push(closeMat[i][j]);

            while (workChildStack.size() != 0) {
              auto childNode = workChildStack.top();
              workChildStack.pop();

              if (mClusterTree(childNode, 0) != 0) {
                workChildStack.push(mClusterTree(childNode, 0));
                workChildStack.push(mClusterTree(childNode, 1));
              } else {
                childCloseMat.push_back(childNode);
              }
            }

            for (size_t k = 0; k < childCloseMat.size(); k++)
              newCloseMat.push_back(childCloseMat[k]);
          }
        }
      }

      closeMat[i] = newCloseMat;

      mLeafNodeList.push_back(node);
    }
  }

  // split leaf node among mpi ranks
  size_t totalCloseEntry = 0;
  size_t totalClosePair = 0;
  std::vector<size_t> closeMatI;
  std::vector<size_t> closeMatJ;
  mMaxCloseDotBlockSize = 0;
  for (size_t i = 0; i < mLeafNodeList.size(); i++) {
    const std::size_t nodeI = mLeafNodeList[i];
    const std::size_t nodeSizeI =
        static_cast<std::size_t>(
            mClusterTree(nodeI, 3) - mClusterTree(nodeI, 2));
    const std::size_t colSize = closeMat[mLeafNodeList[i]].size();
    for (std::size_t j = 0; j < colSize; j++) {
      const std::size_t nodeJ =
          static_cast<std::size_t>(closeMat[nodeI][j]);
      const std::size_t nodeSizeJ =
          static_cast<std::size_t>(
              mClusterTree(nodeJ, 3) - mClusterTree(nodeJ, 2));
      const std::size_t blockEntries =
          CheckedSizeProduct(
              nodeSizeI,
              nodeSizeJ,
              "close block entry count");

      // consider the symmetry property
      if (mUseSymmetry) {
        if (nodeJ >= nodeI) {
          if (totalClosePair % (size_t)mMPISize == (size_t)mMPIRank) {
            closeMatI.push_back(nodeI);
            closeMatJ.push_back(nodeJ);

            if (nodeI == nodeJ)
              totalCloseEntry =
                  CheckedSizeSum(
                      totalCloseEntry,
                      blockEntries,
                      "total close entry count");
            else
              totalCloseEntry =
                  CheckedSizeSum(
                      totalCloseEntry,
                      CheckedSizeProduct(
                          static_cast<std::size_t>(2),
                          blockEntries,
                          "symmetric close block entry count"),
                      "total close entry count");

            if (blockEntries >
                static_cast<std::size_t>(mMaxCloseDotBlockSize))
              mMaxCloseDotBlockSize = ClampSizeToInt(blockEntries);
          }
        }
      } else {
        if (totalClosePair % (size_t)mMPISize == (size_t)mMPIRank) {
          closeMatI.push_back(nodeI);
          closeMatJ.push_back(nodeJ);

          totalCloseEntry =
              CheckedSizeSum(
                  totalCloseEntry,
                  blockEntries,
                  "total close entry count");

          if (blockEntries >
              static_cast<std::size_t>(mMaxCloseDotBlockSize))
            mMaxCloseDotBlockSize = ClampSizeToInt(blockEntries);
        }
      }

      totalClosePair++;
    }
  }

  mCloseMatIPtr =
      std::make_shared<DeviceIndexVector>("mCloseMatI", closeMatI.size());
  mCloseMatJPtr =
      std::make_shared<DeviceIndexVector>("mCloseMatJ", closeMatJ.size());
  auto &mCloseMatI = *mCloseMatIPtr;
  auto &mCloseMatJ = *mCloseMatJPtr;
  DeviceIndexVector::HostMirror hostCloseMatI =
      Kokkos::create_mirror_view(*mCloseMatIPtr);
  DeviceIndexVector::HostMirror hostCloseMatJ =
      Kokkos::create_mirror_view(*mCloseMatJPtr);

  {
    for (size_t i = 0; i < closeMatI.size(); i++) {
      hostCloseMatI(i) = closeMatI[i];
      hostCloseMatJ(i) = closeMatJ[i];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &totalCloseEntry, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);

  if (mMPIRank == 0)
    std::cout << "Total close pair: " << totalClosePair << std::endl;

  if (mMPIRank == 0)
    std::cout << "Total close entry: " << totalCloseEntry << std::endl;

  Kokkos::deep_copy(mCloseMatI, hostCloseMatI);
  Kokkos::deep_copy(mCloseMatJ, hostCloseMatJ);

  std::size_t totalFarSize = 0;
  std::size_t totalFarNode = 0;
  for (size_t i = 0; i < farMat.size(); i++) {
    if (farMat[i].size() != 0) {
      totalFarSize += farMat[i].size();
      totalFarNode++;
    }
  }
  // if (mMPIRank == 0) {
  //   std::cout << "Total far node: " << totalFarNode << std::endl;
  //   std::cout << "Total far pair: " << totalFarSize << std::endl;
  // }
  if (mMPIRank == 0) {
    const std::size_t admissibleBlocks = totalFarSize;
    const std::size_t inadmissibleBlocks = totalClosePair;
    const std::size_t totalBlocks = admissibleBlocks + inadmissibleBlocks;

    std::cout << "Admissible blocks (far pairs): " << admissibleBlocks << std::endl;
    std::cout << "Inadmissible blocks (close pairs): " << inadmissibleBlocks << std::endl;
    std::cout << "Total blocks: " << totalBlocks << std::endl;

    std::cout << "Total far node: " << totalFarNode << std::endl;
    std::cout << "Total far pair: " << totalFarSize << std::endl;
  }

  // split far pair among mpi ranks
  std::vector<size_t> farMatI;
  std::vector<size_t> farMatJ;

  int counter = 0;
  for (size_t i = 0; i < farMat.size(); i++) {
    int nodeI = i;
    for (size_t j = 0; j < farMat[i].size(); j++) {
      int nodeJ = farMat[i][j];

      // consider the symmetry property
      if (mUseSymmetry) {
        if (nodeJ >= nodeI) {
          if (counter % (std::size_t)mMPISize == (std::size_t)mMPIRank) {
            farMatI.push_back(nodeI);
            farMatJ.push_back(nodeJ);
          }
          counter++;
        }
      } else {
        if (counter % (std::size_t)mMPISize == (std::size_t)mMPIRank) {
          farMatI.push_back(nodeI);
          farMatJ.push_back(nodeJ);
        }
        counter++;
      }
    }
  }

  mFarMatIPtr = std::make_shared<DeviceIndexVector>("mFarMatI", farMatI.size());
  auto &mFarMatI = *mFarMatIPtr;

  mFarMatJPtr = std::make_shared<DeviceIndexVector>("mFarMatJ", farMatJ.size());
  auto &mFarMatJ = *mFarMatJPtr;

  DeviceIndexVector::HostMirror farMatIMirror =
      Kokkos::create_mirror_view(mFarMatI);
  DeviceIndexVector::HostMirror farMatJMirror =
      Kokkos::create_mirror_view(mFarMatJ);

  for (size_t i = 0; i < farMatI.size(); i++) {
    farMatIMirror(i) = farMatI[i];
    farMatJMirror(i) = farMatJ[i];
  }

  std::size_t farDotQueryNum = 0;
  std::size_t maxSingleNodeSize = 0;
  for (size_t i = 0; i < farMatIMirror.extent(0); i++) {
    std::size_t nodeISize =
        mClusterTree(farMatIMirror(i), 3) - mClusterTree(farMatIMirror(i), 2);
    std::size_t nodeJSize =
        mClusterTree(farMatJMirror(i), 3) - mClusterTree(farMatJMirror(i), 2);
    const std::size_t blockEntries =
        CheckedSizeProduct(
            nodeISize,
            nodeJSize,
            "far block query count");
    if (mUseSymmetry)
      farDotQueryNum =
          CheckedSizeSum(
              farDotQueryNum,
              CheckedSizeProduct(
                  static_cast<std::size_t>(2),
                  blockEntries,
                  "symmetric far block query count"),
              "total far query count");
    else
      farDotQueryNum =
          CheckedSizeSum(
              farDotQueryNum,
              blockEntries,
              "total far query count");

    if (nodeISize > maxSingleNodeSize)
      maxSingleNodeSize = nodeISize;
    if (nodeJSize > maxSingleNodeSize)
      maxSingleNodeSize = nodeJSize;
  }
  MPI_Allreduce(MPI_IN_PLACE, &farDotQueryNum, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  if (mMPIRank == 0)
    std::cout << "Total far entry: " << farDotQueryNum << std::endl;

  MPI_Allreduce(MPI_IN_PLACE, &maxSingleNodeSize, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                MPI_COMM_WORLD);
  if (mMPIRank == 0)
    std::cout << "Max single node size: " << maxSingleNodeSize << std::endl;

  Kokkos::deep_copy(mFarMatI, farMatIMirror);
  Kokkos::deep_copy(mFarMatJ, farMatJMirror);

  MPI_Barrier(MPI_COMM_WORLD);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  if (mMPIRank == 0) {
    std::cout << "Time for building close and far matrix: "
              << (double)duration / 1e6 << "s" << std::endl;
  }
}
