#include "preprocessing/SFMFrontend.h"

int main(int argc, char **argv) {
  pre::SFMFrontend sfmFrontend(std::string(RESOURCE_DIR) +
                               "/room/calibration_result.yml");
  sfmFrontend.populateImageGraph(std::string(RESOURCE_DIR) + "/room/room_");
  sfmFrontend.processImageGraph();
  sfmFrontend.populateEdges(100);
  // sfmFrontend.printGraphAsMatrix();
  sfmFrontend.incrementalSFM();

  return 0;
}