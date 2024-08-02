// headers in STL
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
// headers in 3rd-part
#include "../pointpillars/pointpillars.h"
#include "gtest/gtest.h"
using namespace std;

#define LOGPF(format, ...) fprintf(stderr ,"[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)

int Txt2Arrary( float* &points_array , string file_name , int num_feature = 4)
{
  ifstream InFile;
  InFile.open(file_name.data());
  assert(InFile.is_open());

  vector<float> temp_points;
  string c;

  uint32_t counter = 0;
  while (!InFile.eof())
  {
      InFile >> c;
      temp_points.push_back(atof(c.c_str()));
      // LOGPF("c: %s, temp_points size: %ld", c.c_str(), temp_points.size());
  }
  LOGPF("temp_points size: %ld", temp_points.size());
  points_array = new float[temp_points.size()];
  for (int i = 0 ; i < temp_points.size() ; ++i) {
    counter += 1;
    if(counter % 5 == 0)
    {
      continue;
    }
    points_array[i] = temp_points[i];
  }

  InFile.close();  
  return temp_points.size() / num_feature;
  // printf("Done");
};

void Boxes2Txt( std::vector<float> boxes , string file_name , int num_feature = 7)
{
    ofstream ofFile;
    ofFile.open(file_name , std::ios::out );  
    if (ofFile.is_open()) {
        for (int i = 0 ; i < boxes.size() / num_feature ; ++i) {
            for (int j = 0 ; j < num_feature ; ++j) {
                ofFile << boxes.at(i * num_feature + j) << " ";
            }
            ofFile << "\n";
        }
    }
    ofFile.close();
    return ;
};

TEST(PointPillars, __build_model__) {

  const std::string DB_CONF = "../bootstrap.yaml";
  YAML::Node config = YAML::LoadFile(DB_CONF);

  std::string pfe_file,backbone_file; 
  if(config["UseOnnx"].as<bool>()) {
    pfe_file = config["PfeOnnx"].as<std::string>();
    backbone_file = config["BackboneOnnx"].as<std::string>();
  }else {
    pfe_file = config["PfeTrt"].as<std::string>();
    backbone_file = config["BackboneTrt"].as<std::string>();
  }
  std::cout << backbone_file << std::endl;
  const std::string pp_config = config["ModelConfig"].as<std::string>();
  PointPillars pp(
    config["ScoreThreshold"].as<float>(),
    config["NmsOverlapThreshold"].as<float>(),
    config["UseOnnx"].as<bool>(),
    pfe_file,
    backbone_file,
    pp_config
  );
  std::string file_name = config["InputFile"].as<std::string>();
  float* points_array;
  int in_num_points;
  // in_num_points = Txt2Arrary(points_array, file_name, 5);
  in_num_points = Txt2Arrary(points_array, file_name, 4);
  LOGPF("in_num_points: %d", in_num_points);

  for (int i = 0 ; i < 10 ; i++)
  {
    std::vector<float> out_detections;
    std::vector<int> out_labels;
    std::vector<float> out_scores;

    cudaDeviceSynchronize();
    pp.DoInference(points_array, in_num_points, &out_detections, &out_labels , &out_scores);
    cudaDeviceSynchronize();
    int BoxFeature = 7;
    int num_objects = out_detections.size() / BoxFeature;

    std::string boxes_file_name = config["OutputFile"].as<std::string>();
    Boxes2Txt(out_detections , boxes_file_name );

    LOGPF("test [%d], num_objects: %d", i, num_objects);
    for(int j=0; j<num_objects; j++)
    {
      LOGPF("obj[%d] wlh(%.2f, %.2f, %.2f), label: %d, score: %.2f", \
            j, out_detections[j*BoxFeature+3], out_detections[j*BoxFeature+4], out_detections[j*BoxFeature+5], \
            out_labels[j], out_scores[j]);
    }
    // EXPECT_EQ(num_objects,226);
  }


};
