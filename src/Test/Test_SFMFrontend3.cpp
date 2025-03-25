#include "preprocessing/SFMFrontend.h"

int main(int argc, char **argv) {
  pre::SFMFrontend sfmFrontend(std::string(RESOURCE_DIR) +
                               "/room/calibration_result.yml");
  sfmFrontend.populateImageGraph(std::string(RESOURCE_DIR) + "/room/room_");
  sfmFrontend.processImageGraph();
  sfmFrontend.populateEdges(100);
  sfmFrontend.printGraphAsMatrix();

  int i = 0, j = 0;
  sfmFrontend.getEdges(i, j);
  sfmFrontend.getEdges(i, j);
  sfmFrontend.printGraphAsMatrix();
  return 0;
}