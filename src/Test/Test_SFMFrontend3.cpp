#include "preprocessing/SFMFrontend.h"

int main(int argc, char **argv) {
  pre::SFMFrontend sfmFrontend(std::string(RESOURCE_DIR) +
                               "/room/calibration_result.yml");
  sfmFrontend.populateImageGraph(std::string(RESOURCE_DIR) + "/room/room_");
  sfmFrontend.processImageGraph(0.6f);
  sfmFrontend.populateEdges(40);
  // sfmFrontend.printGraphAsMatrix();
  // sfmFrontend.incrementalSFM2();
  sfmFrontend.shunxuSFM(std::string(RESOURCE_DIR) + "/room/room_");
  return 0;
}